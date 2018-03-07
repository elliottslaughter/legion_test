/* Copyright 2018 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "legion.h"
#include "legion/legion_ops.h"
#include "legion/legion_spy.h"
#include "legion/legion_trace.h"
#include "legion/legion_tasks.h"
#include "legion/legion_instances.h"
#include "legion/legion_views.h"
#include "legion/legion_context.h"

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // LegionTrace 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LegionTrace::LegionTrace(TaskContext *c, bool logical_only)
      : ctx(c), state(LOGICAL_ONLY), last_memoized(0)
    //--------------------------------------------------------------------------
    {
      physical_trace = logical_only ? NULL
                                    : new PhysicalTrace(c->owner_task->runtime);
    }

    //--------------------------------------------------------------------------
    LegionTrace::~LegionTrace(void)
    //--------------------------------------------------------------------------
    {
      if (physical_trace != NULL)
        delete physical_trace;
    }

    //--------------------------------------------------------------------------
    void LegionTrace::register_physical_only(Operation *op, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      std::pair<Operation*,GenerationID> key(op,gen);
      const unsigned index = operations.size();
      op->set_trace_local_id(index);
      operations.push_back(key);
    }

    //--------------------------------------------------------------------------
    void LegionTrace::replay_aliased_children(
                             std::vector<RegionTreePath> &privilege_paths) const
    //--------------------------------------------------------------------------
    {
      unsigned index = operations.size() - 1;
      std::map<unsigned,LegionVector<AliasChildren>::aligned>::const_iterator
        finder = aliased_children.find(index);
      if (finder == aliased_children.end())
        return;
      for (LegionVector<AliasChildren>::aligned::const_iterator it = 
            finder->second.begin(); it != finder->second.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->req_index < privilege_paths.size());
#endif
        privilege_paths[it->req_index].record_aliased_children(it->depth,
                                                               it->mask);
      }
    }

    //--------------------------------------------------------------------------
    void LegionTrace::end_trace_execution(FenceOp *op)
    //--------------------------------------------------------------------------
    {
      // Register for this fence on every one of the operations in
      // the trace and then clear out the operations data structure
      for (std::set<std::pair<Operation*,GenerationID> >::iterator it =
           frontiers.begin(); it != frontiers.end(); ++it)
      {
        const std::pair<Operation*,GenerationID> &target = *it;
#ifdef DEBUG_LEGION
        assert(!target.first->is_internal_op());
#endif
        op->register_dependence(target.first, target.second);
#ifdef LEGION_SPY
        for (unsigned req_idx = 0; req_idx < num_regions[target]; req_idx++)
        {
          LegionSpy::log_mapping_dependence(
              op->get_context()->get_unique_id(), current_uids[target], req_idx,
              op->get_unique_op_id(), 0, TRUE_DEPENDENCE);
        }
#endif
        // Remove any mapping references that we hold
        target.first->remove_mapping_reference(target.second);
      }
      operations.clear();
      last_memoized = 0;
      frontiers.clear();
#ifdef LEGION_SPY
      current_uids.clear();
      num_regions.clear();
#endif
    }

#ifdef LEGION_SPY
    //--------------------------------------------------------------------------
    UniqueID LegionTrace::get_current_uid_by_index(unsigned op_idx) const
    //--------------------------------------------------------------------------
    {
      assert(op_idx < operations.size());
      const std::pair<Operation*,GenerationID> &key = operations[op_idx];
      std::map<std::pair<Operation*,GenerationID>,UniqueID>::const_iterator
        finder = current_uids.find(key);
      assert(finder != current_uids.end());
      return finder->second;
    }
#endif

    //--------------------------------------------------------------------------
    void LegionTrace::invalidate_trace_cache(void)
    //--------------------------------------------------------------------------
    {
      if (physical_trace != NULL)
        physical_trace->clear_cached_template();
    }

    //--------------------------------------------------------------------------
    void LegionTrace::invalidate_current_template(void)
    //--------------------------------------------------------------------------
    {
      if (physical_trace != NULL)
        physical_trace->invalidate_current_template();
    }

    /////////////////////////////////////////////////////////////
    // StaticTrace
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    StaticTrace::StaticTrace(TaskContext *c,const std::set<RegionTreeID> *trees)
      : LegionTrace(c, true)
    //--------------------------------------------------------------------------
    {
      if (trees != NULL)
        application_trees.insert(trees->begin(), trees->end());
    }
    
    //--------------------------------------------------------------------------
    StaticTrace::StaticTrace(const StaticTrace &rhs)
      : LegionTrace(NULL, true)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    StaticTrace::~StaticTrace(void)
    //--------------------------------------------------------------------------
    {
      // Remove our mapping references and then clear the operations
      for (std::vector<std::pair<Operation*,GenerationID> >::const_iterator it =
            operations.begin(); it != operations.end(); it++)
        it->first->remove_mapping_reference(it->second);
      operations.clear();
    }

    //--------------------------------------------------------------------------
    StaticTrace& StaticTrace::operator=(const StaticTrace &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool StaticTrace::is_fixed(void) const
    //--------------------------------------------------------------------------
    {
      // Static traces are always fixed
      return true;
    }

    //--------------------------------------------------------------------------
    bool StaticTrace::handles_region_tree(RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      if (application_trees.empty())
        return true;
      return (application_trees.find(tid) != application_trees.end());
    }

    //--------------------------------------------------------------------------
    void StaticTrace::record_static_dependences(Operation *op,
                               const std::vector<StaticDependence> *dependences)
    //--------------------------------------------------------------------------
    {
      // Internal operations get to skip this
      if (op->is_internal_op())
        return;
      // All other operations have to add something to the list
      if (dependences == NULL)
        static_dependences.resize(static_dependences.size() + 1);
      else // Add it to the list of static dependences
        static_dependences.push_back(*dependences);
    }

    //--------------------------------------------------------------------------
    void StaticTrace::register_operation(Operation *op, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      std::pair<Operation*,GenerationID> key(op,gen);
      const unsigned index = operations.size();
      if (!implicit_runtime->no_physical_tracing &&
          op->is_memoizing() && !op->is_internal_op())
      {
        if (index != last_memoized)
          REPORT_LEGION_ERROR(ERROR_INCOMPLETE_PHYSICAL_TRACING,
              "Invalid memoization request. A trace cannot be partially "
              "memoized. Please change the mapper to request memoization "
              "for all the operations in the trace");
        op->set_trace_local_id(index);
        last_memoized = index + 1;
      }

      if (!op->is_internal_op())
      {
        frontiers.insert(key);
        const LegionVector<DependenceRecord>::aligned &deps = 
          translate_dependence_records(op, index); 
        operations.push_back(key);
#ifdef LEGION_SPY
        current_uids[key] = op->get_unique_op_id();
        num_regions[key] = op->get_region_count();
#endif
        // Add a mapping reference since people will be 
        // registering dependences
        op->add_mapping_reference(gen);  
        // Then compute all the dependences on this operation from
        // our previous recording of the trace
        for (LegionVector<DependenceRecord>::aligned::const_iterator it = 
              deps.begin(); it != deps.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert((it->operation_idx >= 0) &&
                 ((size_t)it->operation_idx < operations.size()));
#endif
          const std::pair<Operation*,GenerationID> &target = 
                                                operations[it->operation_idx];
          std::set<std::pair<Operation*,GenerationID> >::iterator finder =
            frontiers.find(target);
          if (finder != frontiers.end())
          {
            finder->first->remove_mapping_reference(finder->second);
            frontiers.erase(finder);
          }

          if ((it->prev_idx == -1) || (it->next_idx == -1))
          {
            op->register_dependence(target.first, target.second);
#ifdef LEGION_SPY
            LegionSpy::log_mapping_dependence(
                op->get_context()->get_unique_id(),
                get_current_uid_by_index(it->operation_idx),
                (it->prev_idx == -1) ? 0 : it->prev_idx,
                op->get_unique_op_id(), 
                (it->next_idx == -1) ? 0 : it->next_idx, TRUE_DEPENDENCE);
#endif
          }
          else
          {
            op->register_region_dependence(it->next_idx, target.first,
                                           target.second, it->prev_idx,
                                           it->dtype, it->validates,
                                           it->dependent_mask);
#ifdef LEGION_SPY
            LegionSpy::log_mapping_dependence(
                op->get_context()->get_unique_id(),
                get_current_uid_by_index(it->operation_idx), it->prev_idx,
                op->get_unique_op_id(), it->next_idx, it->dtype);
#endif
          }
        }
      }
      else
      {
        // We already added our creator to the list of operations
        // so the set of dependences is index-1
#ifdef DEBUG_LEGION
        assert(index > 0);
#endif
        const LegionVector<DependenceRecord>::aligned &deps = 
          translate_dependence_records(operations[index-1].first, index-1);
        // Special case for internal operations
        // Internal operations need to register transitive dependences
        // on all the other operations with which it interferes.
        // We can get this from the set of operations on which the
        // operation we are currently performing dependence analysis
        // has dependences.
        InternalOp *internal_op = static_cast<InternalOp*>(op);
#ifdef DEBUG_LEGION
        assert(internal_op == dynamic_cast<InternalOp*>(op));
#endif
        int internal_index = internal_op->get_internal_index();
        for (LegionVector<DependenceRecord>::aligned::const_iterator it = 
              deps.begin(); it != deps.end(); it++)
        {
          // We only record dependences for this internal operation on
          // the indexes for which this internal operation is being done
          if (internal_index != it->next_idx)
            continue;
#ifdef DEBUG_LEGION
          assert((it->operation_idx >= 0) &&
                 ((size_t)it->operation_idx < operations.size()));
#endif
          const std::pair<Operation*,GenerationID> &target = 
                                                operations[it->operation_idx];
          // If this is the case we can do the normal registration
          if ((it->prev_idx == -1) || (it->next_idx == -1))
          {
            internal_op->register_dependence(target.first, target.second);
#ifdef LEGION_SPY
            LegionSpy::log_mapping_dependence(
                op->get_context()->get_unique_id(),
                get_current_uid_by_index(it->operation_idx),
                (it->prev_idx == -1) ? 0 : it->prev_idx,
                op->get_unique_op_id(), 
                (it->next_idx == -1) ? 0 : it->next_idx, TRUE_DEPENDENCE);
#endif
          }
          else
          {
            internal_op->record_trace_dependence(target.first, target.second,
                                               it->prev_idx, it->next_idx,
                                               it->dtype, it->dependent_mask);
#ifdef LEGION_SPY
            LegionSpy::log_mapping_dependence(
                internal_op->get_context()->get_unique_id(),
                get_current_uid_by_index(it->operation_idx), it->prev_idx,
                internal_op->get_unique_op_id(), 0, it->dtype);
#endif
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void StaticTrace::record_dependence(
                                     Operation *target, GenerationID target_gen,
                                     Operation *source, GenerationID source_gen)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }
    
    //--------------------------------------------------------------------------
    void StaticTrace::record_region_dependence(
                                    Operation *target, GenerationID target_gen,
                                    Operation *source, GenerationID source_gen,
                                    unsigned target_idx, unsigned source_idx,
                                    DependenceType dtype, bool validates,
                                    const FieldMask &dependent_mask)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void StaticTrace::record_aliased_children(unsigned req_index,unsigned depth,
                                              const FieldMask &aliase_mask)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    const LegionVector<LegionTrace::DependenceRecord>::aligned& 
        StaticTrace::translate_dependence_records(Operation *op, unsigned index)
    //--------------------------------------------------------------------------
    {
      // If we already translated it then we are done
      if (index < translated_deps.size())
        return translated_deps[index];
      const unsigned start_idx = translated_deps.size();
      translated_deps.resize(index+1);
      RegionTreeForest *forest = ctx->runtime->forest;
      for (unsigned op_idx = start_idx; op_idx <= index; op_idx++)
      {
        const std::vector<StaticDependence> &static_deps = 
          static_dependences[op_idx];
        LegionVector<DependenceRecord>::aligned &translation = 
          translated_deps[op_idx];
        for (std::vector<StaticDependence>::const_iterator it = 
              static_deps.begin(); it != static_deps.end(); it++)
        {
          // Convert the previous offset into an absoluate offset    
          // If the previous offset is larger than the index then 
          // this dependence doesn't matter
          if (it->previous_offset > index)
            continue;
          // Compute the field mask by getting the parent region requirement
          unsigned parent_index = op->find_parent_index(it->current_req_index);
          FieldSpace field_space =  
            ctx->find_logical_region(parent_index).get_field_space();
          const FieldMask dependence_mask = 
            forest->get_node(field_space)->get_field_mask(it->dependent_fields);
          translation.push_back(DependenceRecord(index - it->previous_offset, 
                it->previous_req_index, it->current_req_index,
                it->validates, it->dependence_type, dependence_mask));
        }
      }
      return translated_deps[index];
    }

    /////////////////////////////////////////////////////////////
    // DynamicTrace 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DynamicTrace::DynamicTrace(TraceID t, TaskContext *c, bool logical_only)
      : LegionTrace(c, logical_only), tid(t), fixed(false), tracing(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DynamicTrace::DynamicTrace(const DynamicTrace &rhs)
      : LegionTrace(NULL, true), tid(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    DynamicTrace::~DynamicTrace(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DynamicTrace& DynamicTrace::operator=(const DynamicTrace &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void DynamicTrace::fix_trace(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!fixed);
#endif
      fixed = true;
    }

    //--------------------------------------------------------------------------
    void DynamicTrace::end_trace_capture(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tracing);
#endif
      operations.clear();
      last_memoized = 0;
      op_map.clear();
      internal_dependences.clear();
      tracing = false;
#ifdef LEGION_SPY
      current_uids.clear();
      num_regions.clear();
#endif
    } 

    //--------------------------------------------------------------------------
    bool DynamicTrace::handles_region_tree(RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      // Always handles all of them
      return true;
    }

    //--------------------------------------------------------------------------
    void DynamicTrace::record_static_dependences(Operation *op,
                               const std::vector<StaticDependence> *dependences)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void DynamicTrace::register_operation(Operation *op, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      std::pair<Operation*,GenerationID> key(op,gen);
      const unsigned index = operations.size();
      if (!implicit_runtime->no_physical_tracing &&
          op->is_memoizing() && !op->is_internal_op())
      {
        if (index != last_memoized)
          REPORT_LEGION_ERROR(ERROR_INCOMPLETE_PHYSICAL_TRACING,
              "Invalid memoization request. A trace cannot be partially "
              "memoized. Please change the mapper to request memoization "
              "for all the operations in the trace");
        op->set_trace_local_id(index);
        last_memoized = index + 1;
      }

      // Only need to save this in the map if we are not done tracing
      if (tracing)
      {
        // This is the normal case
        if (!op->is_internal_op())
        {
          operations.push_back(key);
          op_map[key] = index;
          // Add a new vector for storing dependences onto the back
          dependences.push_back(LegionVector<DependenceRecord>::aligned());
          // Record meta-data about the trace for verifying that
          // it is being replayed correctly
          op_info.push_back(OperationInfo(op));
        }
        else // Otherwise, track internal operations separately
        {
          std::pair<InternalOp*,GenerationID> 
            local_key(static_cast<InternalOp*>(op),gen);
          internal_dependences[local_key] = 
            LegionVector<DependenceRecord>::aligned();
        }
      }
      else
      {
        if (!op->is_internal_op())
        {
          frontiers.insert(key);
          // Check for exceeding the trace size
          if (index >= dependences.size())
            REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_RECORDED,
                          "Trace violation! Recorded %zd operations in trace "
                          "%d in task %s (UID %lld) but %d operations have "
                          "now been issued!", dependences.size(), tid,
                          ctx->get_task_name(), ctx->get_unique_id(), index+1)
          // Check to see if the meta-data alignes
          const OperationInfo &info = op_info[index];
          // Check that they are the same kind of operation
          if (info.kind != op->get_operation_kind())
            REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_OPERATION,
                          "Trace violation! Operation at index %d of trace %d "
                          "in task %s (UID %lld) was recorded as having type "
                          "%s but instead has type %s in replay.",
                          index, tid, ctx->get_task_name(),ctx->get_unique_id(),
                          Operation::get_string_rep(info.kind),
                          Operation::get_string_rep(op->get_operation_kind()))
          // Check that they have the same number of region requirements
          if (info.count != op->get_region_count())
            REPORT_LEGION_ERROR(ERROR_TRACE_VIOLATION_OPERATION,
                          "Trace violation! Operation at index %d of trace %d "
                          "in task %s (UID %lld) was recorded as having %d "
                          "regions, but instead has %zd regions in replay.",
                          index, tid, ctx->get_task_name(),
                          ctx->get_unique_id(), info.count,
                          op->get_region_count())
          // If we make it here, everything is good
          const LegionVector<DependenceRecord>::aligned &deps = 
                                                          dependences[index];
          operations.push_back(key);
#ifdef LEGION_SPY
          current_uids[key] = op->get_unique_op_id();
          num_regions[key] = op->get_region_count();
#endif
          // Add a mapping reference since people will be 
          // registering dependences
          op->add_mapping_reference(gen);  
          // Then compute all the dependences on this operation from
          // our previous recording of the trace
          for (LegionVector<DependenceRecord>::aligned::const_iterator it = 
                deps.begin(); it != deps.end(); it++)
          {
#ifdef DEBUG_LEGION
            assert((it->operation_idx >= 0) &&
                   ((size_t)it->operation_idx < operations.size()));
#endif
            const std::pair<Operation*,GenerationID> &target = 
                                                  operations[it->operation_idx];
            std::set<std::pair<Operation*,GenerationID> >::iterator finder =
              frontiers.find(target);
            if (finder != frontiers.end())
            {
              finder->first->remove_mapping_reference(finder->second);
              frontiers.erase(finder);
            }

            if ((it->prev_idx == -1) || (it->next_idx == -1))
            {
              op->register_dependence(target.first, target.second);
#ifdef LEGION_SPY
              LegionSpy::log_mapping_dependence(
                  op->get_context()->get_unique_id(),
                  get_current_uid_by_index(it->operation_idx),
                  (it->prev_idx == -1) ? 0 : it->prev_idx,
                  op->get_unique_op_id(), 
                  (it->next_idx == -1) ? 0 : it->next_idx, TRUE_DEPENDENCE);
#endif
            }
            else
            {
              op->register_region_dependence(it->next_idx, target.first,
                                             target.second, it->prev_idx,
                                             it->dtype, it->validates,
                                             it->dependent_mask);
#ifdef LEGION_SPY
              LegionSpy::log_mapping_dependence(
                  op->get_context()->get_unique_id(),
                  get_current_uid_by_index(it->operation_idx), it->prev_idx,
                  op->get_unique_op_id(), it->next_idx, it->dtype);
#endif
            }
          }
        }
        else
        {
          // We already added our creator to the list of operations
          // so the set of dependences is index-1
#ifdef DEBUG_LEGION
          assert(index > 0);
#endif
          const LegionVector<DependenceRecord>::aligned &deps = 
                                                        dependences[index-1];
          // Special case for internal operations
          // Internal operations need to register transitive dependences
          // on all the other operations with which it interferes.
          // We can get this from the set of operations on which the
          // operation we are currently performing dependence analysis
          // has dependences.
          InternalOp *internal_op = static_cast<InternalOp*>(op);
#ifdef DEBUG_LEGION
          assert(internal_op == dynamic_cast<InternalOp*>(op));
#endif
          int internal_index = internal_op->get_internal_index();
          for (LegionVector<DependenceRecord>::aligned::const_iterator it = 
                deps.begin(); it != deps.end(); it++)
          {
            // We only record dependences for this internal operation on
            // the indexes for which this internal operation is being done
            if (internal_index != it->next_idx)
              continue;
#ifdef DEBUG_LEGION
            assert((it->operation_idx >= 0) &&
                   ((size_t)it->operation_idx < operations.size()));
#endif
            const std::pair<Operation*,GenerationID> &target = 
                                                  operations[it->operation_idx];

            // If this is the case we can do the normal registration
            if ((it->prev_idx == -1) || (it->next_idx == -1))
            {
              internal_op->register_dependence(target.first, target.second);
#ifdef LEGION_SPY
              LegionSpy::log_mapping_dependence(
                  op->get_context()->get_unique_id(),
                  get_current_uid_by_index(it->operation_idx),
                  (it->prev_idx == -1) ? 0 : it->prev_idx,
                  op->get_unique_op_id(), 
                  (it->next_idx == -1) ? 0 : it->next_idx, TRUE_DEPENDENCE);
#endif
            }
            else
            {
              internal_op->record_trace_dependence(target.first, target.second,
                                                 it->prev_idx, it->next_idx,
                                                 it->dtype, it->dependent_mask);
#ifdef LEGION_SPY
              LegionSpy::log_mapping_dependence(
                  internal_op->get_context()->get_unique_id(),
                  get_current_uid_by_index(it->operation_idx), it->prev_idx,
                  internal_op->get_unique_op_id(), 0, it->dtype);
#endif
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void DynamicTrace::record_dependence(Operation *target,GenerationID tar_gen,
                                         Operation *source,GenerationID src_gen)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tracing);
      if (!source->is_internal_op())
      {
        assert(operations.back().first == source);
        assert(operations.back().second == src_gen);
      }
#endif
      std::pair<Operation*,GenerationID> target_key(target, tar_gen);
      std::map<std::pair<Operation*,GenerationID>,unsigned>::const_iterator
        finder = op_map.find(target_key);
      // We only need to record it if it falls within our trace
      if (finder != op_map.end())
      {
        // Two cases here
        if (!source->is_internal_op())
        {
          // Normal case
          insert_dependence(DependenceRecord(finder->second));
        }
        else
        {
          // Otherwise this is an internal op so record it special
          // Don't record dependences on our creator
          if (target_key != operations.back())
          {
            std::pair<InternalOp*,GenerationID> 
              src_key(static_cast<InternalOp*>(source), src_gen);
#ifdef DEBUG_LEGION
            assert(internal_dependences.find(src_key) != 
                   internal_dependences.end());
#endif
            insert_dependence(src_key, DependenceRecord(finder->second));
          }
        }
      }
      else if (target->is_internal_op())
      {
        // They shouldn't both be internal operations, if they are, then
        // they should be going through the other path that tracks
        // dependences based on specific region requirements
#ifdef DEBUG_LEGION
        assert(!source->is_internal_op());
#endif
        // First check to see if the internal op is one of ours
        std::pair<InternalOp*,GenerationID> 
          local_key(static_cast<InternalOp*>(target),tar_gen);
        std::map<std::pair<InternalOp*,GenerationID>,
                LegionVector<DependenceRecord>::aligned>::const_iterator
          internal_finder = internal_dependences.find(local_key);
        if (internal_finder != internal_dependences.end())
        {
          const LegionVector<DependenceRecord>::aligned &internal_deps = 
                                                        internal_finder->second;
          for (LegionVector<DependenceRecord>::aligned::const_iterator it = 
                internal_deps.begin(); it != internal_deps.end(); it++)
            insert_dependence(DependenceRecord(it->operation_idx)); 
        }
      }
    }

    //--------------------------------------------------------------------------
    void DynamicTrace::record_region_dependence(Operation *target, 
                                                GenerationID tar_gen,
                                                Operation *source, 
                                                GenerationID src_gen,
                                                unsigned target_idx, 
                                                unsigned source_idx,
                                                DependenceType dtype,
                                                bool validates,
                                                const FieldMask &dep_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tracing);
      if (!source->is_internal_op())
      {
        assert(operations.back().first == source);
        assert(operations.back().second == src_gen);
      }
#endif
      std::pair<Operation*,GenerationID> target_key(target, tar_gen);
      std::map<std::pair<Operation*,GenerationID>,unsigned>::const_iterator
        finder = op_map.find(target_key);
      // We only need to record it if it falls within our trace
      if (finder != op_map.end())
      {
        // Two cases here, 
        if (!source->is_internal_op())
        {
          // Normal case
          insert_dependence(DependenceRecord(finder->second, target_idx, 
                                source_idx, validates, dtype, dep_mask));
        }
        else
        {
          // Otherwise this is a internal op so record it special
          // Don't record dependences on our creator
          if (target_key != operations.back())
          { 
            std::pair<InternalOp*,GenerationID> 
              src_key(static_cast<InternalOp*>(source), src_gen);
#ifdef DEBUG_LEGION
            assert(internal_dependences.find(src_key) != 
                   internal_dependences.end());
#endif
            insert_dependence(src_key, 
                DependenceRecord(finder->second, target_idx, source_idx,
                                 validates, dtype, dep_mask));
          }
        }
      }
      else if (target->is_internal_op())
      {
        // First check to see if the internal op is one of ours
        std::pair<InternalOp*,GenerationID> 
          local_key(static_cast<InternalOp*>(target), tar_gen);
        std::map<std::pair<InternalOp*,GenerationID>,
                 LegionVector<DependenceRecord>::aligned>::const_iterator
          internal_finder = internal_dependences.find(local_key);
        if (internal_finder != internal_dependences.end())
        {
          // It is one of ours, so two cases
          if (!source->is_internal_op())
          {
            // Iterate over the internal operation dependences and 
            // translate them to our dependences
            for (LegionVector<DependenceRecord>::aligned::const_iterator
                  it = internal_finder->second.begin(); 
                  it != internal_finder->second.end(); it++)
            {
              FieldMask overlap = it->dependent_mask & dep_mask;
              if (!overlap)
                continue;
              insert_dependence(DependenceRecord(it->operation_idx, 
                  it->prev_idx, source_idx, it->validates, it->dtype, overlap));
            }
          }
          else
          {
            // Iterate over the internal operation dependences
            // and translate them to our dependences
            std::pair<InternalOp*,GenerationID> 
              src_key(static_cast<InternalOp*>(source), src_gen);
#ifdef DEBUG_LEGION
            assert(internal_dependences.find(src_key) != 
                   internal_dependences.end());
#endif
            for (LegionVector<DependenceRecord>::aligned::const_iterator
                  it = internal_finder->second.begin(); 
                  it != internal_finder->second.end(); it++)
            {
              FieldMask overlap = it->dependent_mask & dep_mask;
              if (!overlap)
                continue;
              insert_dependence(src_key, 
                  DependenceRecord(it->operation_idx, it->prev_idx,
                    source_idx, it->validates, it->dtype, overlap));
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void DynamicTrace::record_aliased_children(unsigned req_index,
                                          unsigned depth, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      unsigned index = operations.size() - 1;
      aliased_children[index].push_back(AliasChildren(req_index, depth, mask));
    } 

    //--------------------------------------------------------------------------
    void DynamicTrace::insert_dependence(const DependenceRecord &record)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!dependences.empty());
#endif
      LegionVector<DependenceRecord>::aligned &deps = dependences.back();
      // Try to merge it with an existing dependence
      for (unsigned idx = 0; idx < deps.size(); idx++)
        if (deps[idx].merge(record))
          return;
      // If we make it here, we couldn't merge it so just add it
      deps.push_back(record);
    }

    //--------------------------------------------------------------------------
    void DynamicTrace::insert_dependence(
                                 const std::pair<InternalOp*,GenerationID> &key,
                                 const DependenceRecord &record)
    //--------------------------------------------------------------------------
    {
      LegionVector<DependenceRecord>::aligned &deps = internal_dependences[key];
      // Try to merge it with an existing dependence
      for (unsigned idx = 0; idx < deps.size(); idx++)
        if (deps[idx].merge(record))
          return;
      // If we make it here, we couldn't merge it so just add it
      deps.push_back(record);
    }

    /////////////////////////////////////////////////////////////
    // TraceOp 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceOp::TraceOp(Runtime *rt)
      : FenceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceOp::TraceOp(const TraceOp &rhs)
      : FenceOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TraceOp::~TraceOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceOp& TraceOp::operator=(const TraceOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void TraceOp::execute_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapping_tracker == NULL);
#endif
      // Make a dependence tracker
      mapping_tracker = new MappingDependenceTracker();
      // See if we have any fence dependences
      execution_fence_event = parent_ctx->register_fence_dependence(this);
      parent_ctx->invalidate_trace_cache(local_trace);

      trigger_dependence_analysis();
      end_dependence_analysis();
    }

    /////////////////////////////////////////////////////////////
    // TraceCaptureOp 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceCaptureOp::TraceCaptureOp(Runtime *rt)
      : TraceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceCaptureOp::TraceCaptureOp(const TraceCaptureOp &rhs)
      : TraceOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TraceCaptureOp::~TraceCaptureOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceCaptureOp& TraceCaptureOp::operator=(const TraceCaptureOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void TraceCaptureOp::initialize_capture(TaskContext *ctx)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, MIXED_FENCE);
#ifdef DEBUG_LEGION
      assert(trace != NULL);
      assert(trace->is_dynamic_trace());
#endif
      dynamic_trace = trace->as_dynamic_trace();
      local_trace = dynamic_trace;
      // Now mark our trace as NULL to avoid registering this operation
      trace = NULL;
      tracing = false;
      current_template = NULL;
    }

    //--------------------------------------------------------------------------
    void TraceCaptureOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
    }

    //--------------------------------------------------------------------------
    void TraceCaptureOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      runtime->free_capture_op(this);
    }

    //--------------------------------------------------------------------------
    const char* TraceCaptureOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_CAPTURE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind TraceCaptureOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_CAPTURE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void TraceCaptureOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(trace == NULL);
      assert(local_trace != NULL);
      assert(local_trace == dynamic_trace);
#endif
      // Indicate that we are done capturing this trace
      dynamic_trace->end_trace_capture();
      // Register this fence with all previous users in the parent's context
      FenceOp::trigger_dependence_analysis();
      parent_ctx->record_previous_trace(local_trace);
      if (local_trace->is_recording())
      {
#ifdef DEBUG_LEGION
        assert(local_trace->get_physical_trace() != NULL);
#endif
        current_template =
          local_trace->get_physical_trace()->get_current_template();
      }
    }

    //--------------------------------------------------------------------------
    void TraceCaptureOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Now finish capturing the physical trace
      if (local_trace->is_recording())
      {
#ifdef DEBUG_LEGION
        assert(current_template != NULL);
        assert(local_trace->get_physical_trace() != NULL);
#endif
        local_trace->get_physical_trace()->fix_trace(current_template);
        local_trace->initialize_tracing_state();
      }
      FenceOp::trigger_mapping();
    }

    /////////////////////////////////////////////////////////////
    // TraceCompleteOp 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceCompleteOp::TraceCompleteOp(Runtime *rt)
      : TraceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceCompleteOp::TraceCompleteOp(const TraceCompleteOp &rhs)
      : TraceOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TraceCompleteOp::~TraceCompleteOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceCompleteOp& TraceCompleteOp::operator=(const TraceCompleteOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::initialize_complete(TaskContext *ctx)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, MIXED_FENCE);
#ifdef DEBUG_LEGION
      assert(trace != NULL);
#endif
      local_trace = trace;
      // Now mark our trace as NULL to avoid registering this operation
      trace = NULL;
      current_template = NULL;
      template_completion = ApEvent::NO_AP_EVENT;
      replayed = false;
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      runtime->free_trace_op(this);
    }

    //--------------------------------------------------------------------------
    const char* TraceCompleteOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_COMPLETE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind TraceCompleteOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_COMPLETE_OP_KIND; 
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(trace == NULL);
      assert(local_trace != NULL);
#endif
      if (local_trace->is_replaying())
      {
#ifdef DEBUG_LEGION
        assert(local_trace->get_physical_trace() != NULL);
#endif
        PhysicalTemplate *current_template =
          local_trace->get_physical_trace()->get_current_template();
        template_completion = current_template->get_completion();
        Runtime::trigger_event(completion_event, template_completion);
        local_trace->end_trace_execution(this);
        parent_ctx->update_current_fence(this, true, true);
        parent_ctx->record_previous_trace(local_trace);
        local_trace->initialize_tracing_state();
        replayed = true;
        return;
      }
      else if (local_trace->is_recording())
      {
#ifdef DEBUG_LEGION
        assert(local_trace->get_physical_trace() != NULL);
#endif
        current_template =
          local_trace->get_physical_trace()->get_current_template();
      }

      // Indicate that this trace is done being captured
      // This also registers that we have dependences on all operations
      // in the trace.
      local_trace->end_trace_execution(this);

      // We always need to run the full fence analysis, otherwise
      // the operations replayed in the following trace will race
      // with those in the current trace
      execution_precondition =
        parent_ctx->perform_fence_analysis(this, true, true);

      // Now update the parent context with this fence before we can complete
      // the dependence analysis and possibly be deactivated
      parent_ctx->update_current_fence(this, true, true);

      // If this is a static trace, then we remove our reference when we're done
      if (local_trace->is_static_trace())
      {
        StaticTrace *static_trace = static_cast<StaticTrace*>(local_trace);
        if (static_trace->remove_reference())
          delete static_trace;
      }
      parent_ctx->record_previous_trace(local_trace);
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Now finish capturing the physical trace
      if (local_trace->is_recording())
      {
#ifdef DEBUG_LEGION
        assert(current_template != NULL);
        assert(local_trace->get_physical_trace() != NULL);
#endif
        local_trace->get_physical_trace()->fix_trace(current_template);
        local_trace->initialize_tracing_state();
      }
      else if (replayed)
      {
        complete_mapping();
        need_completion_trigger = false;
        if (!template_completion.has_triggered())
        {
          RtEvent wait_on = Runtime::protect_event(template_completion);
          complete_execution(wait_on);
        }
        else
          complete_execution();
        return;
      }
      FenceOp::trigger_mapping();
    }

    /////////////////////////////////////////////////////////////
    // TraceReplayOp
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceReplayOp::TraceReplayOp(Runtime *rt)
      : TraceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceReplayOp::TraceReplayOp(const TraceReplayOp &rhs)
      : TraceOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TraceReplayOp::~TraceReplayOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceReplayOp& TraceReplayOp::operator=(const TraceReplayOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void TraceReplayOp::initialize_replay(TaskContext *ctx, LegionTrace *trace)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, EXECUTION_FENCE);
#ifdef DEBUG_LEGION
      assert(trace != NULL);
#endif
      local_trace = trace;
      trace = NULL;
    }

    //--------------------------------------------------------------------------
    void TraceReplayOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
    }

    //--------------------------------------------------------------------------
    void TraceReplayOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      runtime->free_replay_op(this);
    }

    //--------------------------------------------------------------------------
    const char* TraceReplayOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_REPLAY_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind TraceReplayOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_REPLAY_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void TraceReplayOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(trace == NULL);
      assert(local_trace != NULL);
#endif
      PhysicalTrace *physical_trace = local_trace->get_physical_trace();
#ifdef DEBUG_LEGION
      assert(physical_trace != NULL);
#endif
      bool recurrent = true;
      bool fence_registered = false;
      bool is_recording = local_trace->is_recording();
      if (physical_trace->get_current_template() == NULL || is_recording)
      {
        recurrent = false;
        if (physical_trace->has_any_templates() || is_recording)
        {
          // Wait for the previous recordings to be done before checking
          // template preconditions, otherwise no template would exist.
          RtEvent mapped_event = parent_ctx->get_current_mapping_fence_event();
          if (mapped_event.exists())
            mapped_event.wait();
        }

        if (physical_trace->get_current_template() == NULL)
          physical_trace->check_template_preconditions();

        // Register this fence with all previous users in the parent's context
#ifdef LEGION_SPY
        execution_precondition = 
          parent_ctx->perform_fence_analysis(this, true, true);
#else
        execution_precondition = 
          parent_ctx->perform_fence_analysis(this, false, true);
#endif
        fence_registered = true;
      }

      if (physical_trace->get_current_template() != NULL)
      {
        physical_trace->initialize_template(get_completion_event(), recurrent);
        local_trace->set_state_replay();
      }
      else if (!fence_registered)
      {
#ifdef LEGION_SPY
        execution_precondition = 
          parent_ctx->perform_fence_analysis(this, true, true);
#else
        execution_precondition = 
          parent_ctx->perform_fence_analysis(this, false, true);
#endif
      }

      // Now update the parent context with this fence before we can complete
      // the dependence analysis and possibly be deactivated
#ifdef LEGION_SPY
      parent_ctx->update_current_fence(this, true, true);
#else
      parent_ctx->update_current_fence(this, false, true);
#endif
    }

    /////////////////////////////////////////////////////////////
    // TraceBeginOp
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceBeginOp::TraceBeginOp(Runtime *rt)
      : TraceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceBeginOp::TraceBeginOp(const TraceBeginOp &rhs)
      : TraceOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TraceBeginOp::~TraceBeginOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TraceBeginOp& TraceBeginOp::operator=(const TraceBeginOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void TraceBeginOp::initialize_begin(TaskContext *ctx, LegionTrace *trace)
    //--------------------------------------------------------------------------
    {
      initialize(ctx, MAPPING_FENCE);
#ifdef DEBUG_LEGION
      assert(trace != NULL);
#endif
      local_trace = trace;
      trace = NULL;
    }

    //--------------------------------------------------------------------------
    void TraceBeginOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
    }

    //--------------------------------------------------------------------------
    void TraceBeginOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      runtime->free_begin_op(this);
    }

    //--------------------------------------------------------------------------
    const char* TraceBeginOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_BEGIN_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind TraceBeginOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_BEGIN_OP_KIND;
    }

    /////////////////////////////////////////////////////////////
    // PhysicalTrace
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalTrace::PhysicalTrace(Runtime *rt)
      : runtime(rt), current_template(NULL), nonreplayable_count(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalTrace::PhysicalTrace(const PhysicalTrace &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalTrace::~PhysicalTrace()
    //--------------------------------------------------------------------------
    {
      for (std::vector<PhysicalTemplate*>::iterator it = templates.begin();
           it != templates.end(); ++it)
        delete (*it);
      templates.clear();
    }

    //--------------------------------------------------------------------------
    PhysicalTrace& PhysicalTrace::operator=(const PhysicalTrace &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::fix_trace(PhysicalTemplate *tpl)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tpl->is_recording());
#endif
      tpl->finalize();
      if (!tpl->is_replayable())
      {
        delete tpl;
        current_template = NULL;
        if (++nonreplayable_count > LEGION_NON_REPLAYABLE_WARNING)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_NON_REPLAYABLE_COUNT_EXCEEDED,
              "WARNING: The runtime has failed to memoize the trace more than "
              "%u times, due to the absence of a replayable template. It is "
              "highly likely that your trace will not be memoized for the rest "
              "of execution. Please change the mapper to stop making "
              "memoization requests.", LEGION_NON_REPLAYABLE_WARNING)
          nonreplayable_count = 0;
        }
      }
      else
        templates.push_back(tpl);
    }
    //--------------------------------------------------------------------------
    void PhysicalTrace::invalidate_current_template(void)
    //--------------------------------------------------------------------------
    {
      if (current_template != NULL && current_template->is_recording())
        current_template->record_blocking_call();
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::check_template_preconditions(void)
    //--------------------------------------------------------------------------
    {
      for (std::vector<PhysicalTemplate*>::reverse_iterator it =
           templates.rbegin(); it != templates.rend(); ++it)
        if ((*it)->check_preconditions())
        {
#ifdef DEBUG_LEGION
          assert((*it)->is_replayable());
#endif
          current_template = *it;
          return;
        }
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate* PhysicalTrace::start_new_template()
    //--------------------------------------------------------------------------
    {
      current_template = new PhysicalTemplate();
      return current_template;
    }

    //--------------------------------------------------------------------------
    void PhysicalTrace::initialize_template(
                                       ApEvent fence_completion, bool recurrent)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_template != NULL);
#endif
      current_template->initialize(fence_completion, recurrent);
    }

    /////////////////////////////////////////////////////////////
    // PhysicalTemplate
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalTemplate::PhysicalTemplate(void)
      : recording(true), replayable(true), has_block(false),
        fence_completion_id(0)
    //--------------------------------------------------------------------------
    {
      events.push_back(ApEvent());
      instructions.push_back(
          new AssignFenceCompletion(*this, fence_completion_id));
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate::~PhysicalTemplate(void)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock tpl_lock(template_lock);
        for (std::vector<Instruction*>::iterator it = instructions.begin();
             it != instructions.end(); ++it)
          delete *it;
        // Relesae references to instances
        for (CachedMappings::iterator it = cached_mappings.begin();
            it != cached_mappings.end(); ++it)
        {
          for (std::deque<InstanceSet>::iterator pit =
              it->second.physical_instances.begin(); pit !=
              it->second.physical_instances.end(); pit++)
            pit->clear();
        }
        cached_mappings.clear();
      }
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate::PhysicalTemplate(const PhysicalTemplate &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::initialize(ApEvent completion, bool recurrent)
    //--------------------------------------------------------------------------
    {
      fence_completion = completion;
      if (recurrent)
        for (std::map<unsigned, unsigned>::iterator it = frontiers.begin();
            it != frontiers.end(); ++it)
        {
          if (events[it->first].exists())
            events[it->second] = events[it->first];
          else
            events[it->second] = completion;
        }
      else
        for (std::map<unsigned, unsigned>::iterator it = frontiers.begin();
            it != frontiers.end(); ++it)
          events[it->second] = completion;

      events[fence_completion_id] = fence_completion;
#ifdef DEBUG_LEGION
      for (std::map<TraceLocalID, Operation*>::iterator it =
           operations.begin(); it != operations.end(); ++it)
        it->second = NULL;
#endif
    }

    //--------------------------------------------------------------------------
    ApEvent PhysicalTemplate::get_completion(void) const
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> to_merge;
      for (std::map<unsigned, unsigned>::const_iterator it = frontiers.begin();
           it != frontiers.end(); ++it)
        to_merge.insert(events[it->first]);
      return Runtime::merge_events(to_merge);
    }

    //--------------------------------------------------------------------------
    /*static*/ bool PhysicalTemplate::check_logical_open(RegionTreeNode *node,
                                                         ContextID ctx,
                                                         FieldMask fields)
    //--------------------------------------------------------------------------
    {
      {
        const LogicalState &state = node->get_logical_state(ctx);
        fields -= state.dirty_fields;
        if (!fields) return true;
      }

      RegionTreeNode *parent_node = node->get_parent();
      if (parent_node != NULL)
      {
#ifdef DEBUG_LEGION
        assert(!parent_node->is_region());
#endif
        const LogicalState &state = parent_node->get_logical_state(ctx);
#ifdef DEBUG_LEGION
        assert(!!fields);
#endif
        for (LegionList<FieldState>::aligned::const_iterator fit =
             state.field_states.begin(); fit !=
             state.field_states.end(); ++fit)
        {
          if (fit->open_state == NOT_OPEN)
            continue;
          FieldMask overlap = fit->valid_fields & fields;
          if (!overlap)
            continue;
          // FIXME: This code will not work as expected if the projection
          //        goes deeper than one level
          const LegionColor &color = node->get_row_source()->color;
          if ((fit->projection != 0 &&
               fit->projection_space->contains_color(color)) ||
              fit->open_children.find(color) != fit->open_children.end())
            fields -= overlap;
        }
      }

      const LogicalState &state = node->get_logical_state(ctx);
      for (LegionList<FieldState>::aligned::const_iterator fit =
           state.field_states.begin(); fit !=
           state.field_states.end(); ++fit)
      {
        if (fit->open_state == NOT_OPEN)
          continue;
        FieldMask overlap = fit->valid_fields & fields;
        if (!overlap)
          continue;
        fields -= overlap;
      }
      return !fields;
    }

    //--------------------------------------------------------------------------
    /*static*/ bool PhysicalTemplate::check_logical_open(RegionTreeNode *node,
                                                         ContextID ctx,
                           LegionMap<IndexSpaceNode*, FieldMask>::aligned projs)
    //--------------------------------------------------------------------------
    {
      const LogicalState &state = node->get_logical_state(ctx);
      for (LegionList<FieldState>::aligned::const_iterator fit =
           state.field_states.begin(); fit !=
           state.field_states.end(); ++fit)
      {
        if (fit->open_state == NOT_OPEN)
          continue;
        if (fit->projection != 0)
        {
          LegionMap<IndexSpaceNode*, FieldMask>::aligned::iterator finder =
            projs.find(fit->projection_space);
          if (finder != projs.end())
          {
            FieldMask overlap = finder->second & fit->valid_fields;
            if (!overlap)
              continue;
            finder->second -= overlap;
            if (!finder->second)
              projs.erase(finder);
          }
        }
      }
      return projs.size() == 0;
    }

    //--------------------------------------------------------------------------
    bool PhysicalTemplate::check_preconditions(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(trace->runtime, PHYSICAL_TRACE_PRECONDITION_CHECK_CALL);
      for (LegionMap<std::pair<RegionTreeNode*, ContextID>,
                     FieldMask>::aligned::iterator it =
           previous_open_nodes.begin(); it !=
           previous_open_nodes.end(); ++it)
        if (!check_logical_open(it->first.first, it->first.second, it->second))
          return false;

      for (std::map<std::pair<RegionTreeNode*, ContextID>,
           LegionMap<IndexSpaceNode*, FieldMask>::aligned>::iterator it =
           previous_projections.begin(); it !=
           previous_projections.end(); ++it)
        if (!check_logical_open(it->first.first, it->first.second, it->second))
          return false;

      for (LegionMap<InstanceView*, FieldMask>::aligned::iterator it =
           previous_valid_views.begin(); it !=
           previous_valid_views.end(); ++it)
      {
#ifdef DEBUG_LEGION
        assert(logical_contexts.find(it->first) != logical_contexts.end());
        assert(physical_contexts.find(it->first) != physical_contexts.end());
#endif
        RegionTreeNode *logical_node = it->first->logical_node;
        ContextID logical_ctx = logical_contexts[it->first];
        std::pair<RegionTreeNode*, ContextID> key(logical_node, logical_ctx);

        if (previous_open_nodes.find(key) == previous_open_nodes.end() &&
            !check_logical_open(logical_node, logical_ctx, it->second))
          return false;

        ContextID physical_ctx = physical_contexts[it->first];
        PhysicalState *state = new PhysicalState(logical_node, false);
        VersionManager &manager =
          logical_node->get_current_version_manager(physical_ctx);
        manager.update_physical_state(state);
        state->capture_state();

        bool found = false;
        if (it->first->is_materialized_view())
        {
          for (LegionMap<LogicalView*, FieldMask,
                         VALID_VIEW_ALLOC>::track_aligned::iterator vit =
               state->valid_views.begin(); vit !=
               state->valid_views.end(); ++vit)
          {
            if (vit->first->is_materialized_view() &&
                it->first == vit->first && !(it->second - vit->second))
            {
              found = true;
              break;
            }
          }
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(it->first->is_reduction_view());
#endif
          for (LegionMap<ReductionView*, FieldMask,
                         VALID_VIEW_ALLOC>::track_aligned::iterator vit =
               state->reduction_views.begin(); vit !=
               state->reduction_views.end(); ++vit)
          {
            if (it->first == vit->first && !(it->second - vit->second))
            {
              found = true;
              break;
            }
          }
        }
        if (!found)
          return false;
      }

      return true;
    }

    //--------------------------------------------------------------------------
    bool PhysicalTemplate::check_replayable(void)
    //--------------------------------------------------------------------------
    {
      if (untracked_fill_views.size() > 0 || has_block)
        return false;
      for (LegionMap<InstanceView*, FieldMask>::aligned::const_iterator it =
           previous_valid_views.begin(); it !=
           previous_valid_views.end(); ++it)
      {
        LegionMap<InstanceView*, FieldMask>::aligned::const_iterator finder =
          valid_views.find(it->first);
        if (finder == valid_views.end() || !!(it->second - finder->second))
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::register_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      Memoizable *memoizable = op->get_memoizable();
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      std::map<TraceLocalID, Operation*>::iterator op_finder =
        operations.find(memoizable->get_trace_local_id());
#ifdef DEBUG_LEGION
      assert(op_finder != operations.end());
      assert(op_finder->second == NULL);
#endif
      op_finder->second = op;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::execute_all(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(task->runtime, PHYSICAL_TRACE_EXECUTE_CALL);
      for (std::vector<Instruction*>::const_iterator it = instructions.begin();
           it != instructions.end(); ++it)
        (*it)->execute();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::execute(Memoizable *memoizable)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      TraceLocalID op_key = memoizable->get_trace_local_id();
      std::map<TraceLocalID,std::vector<Instruction*> >::iterator finder =
        schedules.find(op_key);
#ifdef DEBUG_LEGION
      assert(finder != schedules.end());
      assert(operations.find(op_key) != operations.end());
      assert(operations.find(op_key)->second != NULL);
#endif
      for (std::vector<Instruction*>::const_iterator it =
           finder->second.begin(); it != finder->second.end(); ++it)
        (*it)->execute();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::finalize(void)
    //--------------------------------------------------------------------------
    {
      recording = false;
      replayable = check_replayable();
      if (outstanding_gc_events.size() > 0)
        for (std::map<InstanceView*, std::set<ApEvent> >::iterator it =
             outstanding_gc_events.begin(); it !=
             outstanding_gc_events.end(); ++it)
        {
          it->first->update_gc_events(it->second);
          it->first->collect_users(it->second);
        }
      if (!replayable)
      {
        if (implicit_runtime->dump_physical_traces)
        {
          optimize();
          dump_template();
        }
        return;
      }
      optimize();
      schedule_instructions();
      if (implicit_runtime->dump_physical_traces) dump_template();
      size_t num_events = events.size();
      events.clear();
      events.resize(num_events);
      event_map.clear();
#ifdef DEBUG_LEGION
      sanity_check();
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::optimize(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(trace->runtime, PHYSICAL_TRACE_OPTIMIZE_CALL);
      std::vector<unsigned> generate;
      std::vector<std::set<unsigned> > preconditions;
      std::vector<std::set<unsigned> > simplified;
      std::map<TraceLocalID, std::set<unsigned> > ready_preconditions;
      generate.resize(instructions.size());
      preconditions.resize(instructions.size());
      simplified.resize(instructions.size());

      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
        std::set<unsigned> &pre = preconditions[idx];
        switch (instructions[idx]->get_kind())
        {
          case ASSIGN_FENCE_COMPLETION:
          case GET_TERM_EVENT:
          case CREATE_AP_USER_EVENT:
            {
              generate[idx] = idx;
              break;
            }
          case TRIGGER_EVENT:
            {
              TriggerEvent *inst = instructions[idx]->as_trigger_event();
              unsigned rhs = inst->rhs;
              pre.insert(preconditions[rhs].begin(), preconditions[rhs].end());
              pre.insert(generate[rhs]);
              generate[idx] = idx;
              break;
            }
          case MERGE_EVENT:
            {
              MergeEvent *inst = instructions[idx]->as_merge_event();
              for (std::set<unsigned>::iterator it = inst->rhs.begin();
                   it != inst->rhs.end(); ++it)
              {
                pre.insert(preconditions[*it].begin(),
                           preconditions[*it].end());
                unsigned rhs_gen = generate[*it];
                pre.insert(preconditions[rhs_gen].begin(),
                           preconditions[rhs_gen].end());
              }
              std::set<unsigned> gen;
              for (std::set<unsigned>::iterator it = inst->rhs.begin();
                   it != inst->rhs.end(); ++it)
                if (pre.find(*it) == pre.end()) gen.insert(generate[*it]);
              if (gen.size() > 1)
              {
                pre.insert(gen.begin(), gen.end());
                generate[idx] = idx;

                std::set<unsigned> &simpl = simplified[idx];
                std::set<unsigned> pre_pre;
                for (std::set<unsigned>::iterator it = pre.begin(); it != pre.end();
                    ++it)
                  pre_pre.insert(preconditions[*it].begin(),
                                 preconditions[*it].end());
                for (std::set<unsigned>::iterator it = pre.begin(); it != pre.end();
                    ++it)
                  if (pre_pre.find(*it) == pre_pre.end())
                    simpl.insert(*it);
              }
              else
              {
#ifdef DEBUG_LEGION
                assert(gen.size() == 1);
#endif
                generate[idx] = *gen.begin();
              }
              break;
            }
          case ISSUE_COPY:
            {
              generate[idx] = idx;
              IssueCopy *inst = instructions[idx]->as_issue_copy();
              std::set<unsigned> &pre_pre = preconditions[inst->precondition_idx];
              pre.insert(generate[inst->precondition_idx]);
              pre.insert(pre_pre.begin(), pre_pre.end());
              break;
            }
          case ISSUE_FILL:
            {
              generate[idx] = idx;
              IssueFill *inst = instructions[idx]->as_issue_fill();
              std::set<unsigned> &pre_pre = preconditions[inst->precondition_idx];
              pre.insert(generate[inst->precondition_idx]);
              pre.insert(pre_pre.begin(), pre_pre.end());
              break;
            }
          case SET_READY_EVENT:
            {
              SetReadyEvent *inst = instructions[idx]->as_set_ready_event();
              std::set<unsigned> &ready_pre = preconditions[inst->ready_event_idx];
              {
                std::map<TraceLocalID, unsigned>::iterator finder =
                  task_entries.find(inst->op_key);
#ifdef DEBUG_LEGION
                assert(finder != task_entries.end());
#endif
                std::set<unsigned> &task_pre = preconditions[finder->second];
                task_pre.insert(generate[inst->ready_event_idx]);
                task_pre.insert(ready_pre.begin(), ready_pre.end());
              }
              {
                std::map<TraceLocalID, std::set<unsigned> >::iterator finder =
                  ready_preconditions.find(inst->op_key);
                if (finder == ready_preconditions.end())
                  ready_preconditions[inst->op_key] = ready_pre;
                else
                  finder->second.insert(ready_pre.begin(), ready_pre.end());
              }
              break;
            }
          case GET_COPY_TERM_EVENT:
          case SET_COPY_SYNC_EVENT:
            {
              generate[idx] = idx;
              break;
            }
          case TRIGGER_COPY_COMPLETION:
            {
              generate[idx] = idx;
              TriggerCopyCompletion *inst =
                instructions[idx]->as_trigger_copy_completion();
              std::set<unsigned> &rhs_pre = preconditions[inst->rhs];
              std::map<TraceLocalID, unsigned>::iterator finder =
                task_entries.find(inst->lhs);
#ifdef DEBUG_LEGION
              assert(finder != task_entries.end());
#endif
              std::set<unsigned> &copy_pre = preconditions[finder->second];
              copy_pre.insert(generate[inst->rhs]);
              copy_pre.insert(rhs_pre.begin(), rhs_pre.end());
              break;
            }
          case LAUNCH_TASK:
            {
              assert(false);
              break;
            }
#ifdef DEBUG_LEGION
          default:
            {
              assert(false);
              break;
            }
#endif
        }
      }

      std::vector<Instruction*> new_instructions;
      unsigned count = 0;
      std::map<unsigned, unsigned> rewrite;
      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
#ifdef DEBUG_LEGION
        assert(rewrite.find(idx) == rewrite.end());
        assert(rewrite.size() == idx);
#endif
        switch (instructions[idx]->get_kind())
        {
          case ASSIGN_FENCE_COMPLETION:
          case GET_TERM_EVENT:
          case CREATE_AP_USER_EVENT:
          case TRIGGER_EVENT:
          case ISSUE_COPY:
          case ISSUE_FILL:
          case GET_COPY_TERM_EVENT:
          case SET_COPY_SYNC_EVENT:
          case TRIGGER_COPY_COMPLETION:
            {
#ifdef DEBUG_LEGION
              assert(generate[idx] == idx);
#endif
              rewrite[idx] = count;
              new_instructions.push_back(
                  instructions[idx]->clone(*this, rewrite));
              ++count;
              break;
            }
          case SET_READY_EVENT:
            {
              SetReadyEvent *inst = instructions[idx]->as_set_ready_event();
              rewrite[idx] = count;
              new_instructions.push_back(inst->clone(*this, rewrite));
              ++count;
              break;
            }
          case MERGE_EVENT:
            {
              if (generate[idx] == idx)
              {
                if (simplified[idx].size() > 1)
                {
                  std::set<unsigned> rhs;
                  for (std::set<unsigned>::iterator it =
                       simplified[idx].begin(); it !=
                       simplified[idx].end(); ++it)
                  {
#ifdef DEBUG_LEGION
                    assert(rewrite.find(*it) != rewrite.end());
#endif
                    rhs.insert(rewrite[*it]);
                  }
                  rewrite[idx] = count;
                  new_instructions.push_back(new MergeEvent(*this, count, rhs));
                  ++count;
                }
                else
                  rewrite[idx] = rewrite[*simplified[idx].begin()];
              }
              else
                rewrite[idx] = rewrite[generate[idx]];
              break;
            }
          case LAUNCH_TASK:
            {
              assert(false);
              break;
            }
#ifdef DEBUG_LEGION
          default:
            {
              assert(false);
              break;
            }
#endif
        }
      }

      instructions.swap(new_instructions);

      for (std::vector<Instruction*>::iterator it = new_instructions.begin();
           it != new_instructions.end(); ++it)
        delete (*it);

      size_t num_origins = 0;
      for (std::vector<Instruction*>::iterator it = instructions.begin();
           it != instructions.end(); ++it)
        switch ((*it)->get_kind())
        {
          case ISSUE_COPY:
            {
              num_origins +=
                (*it)->as_issue_copy()->precondition_idx == fence_completion_id;
              break;
            }
          case ISSUE_FILL:
            {
              num_origins +=
                (*it)->as_issue_fill()->precondition_idx == fence_completion_id;
              break;
            }
          case SET_READY_EVENT:
            {
              num_origins +=
                (*it)->as_set_ready_event()->ready_event_idx ==
                fence_completion_id;
              break;
            }
          default:
            {
              break;
            }
        }
      events.resize(instructions.size() + num_origins);
      std::map<InstanceAccess, UserInfo> new_last_users;
      for (std::map<InstanceAccess, UserInfo>::iterator it = last_users.begin();
           it != last_users.end(); ++it)
      {
        std::set<unsigned> pre;
        for (std::set<unsigned>::iterator uit = it->second.users.begin(); uit !=
             it->second.users.end(); ++uit)
          pre.insert(preconditions[*uit].begin(), preconditions[*uit].end());
        for (std::set<unsigned>::iterator uit = it->second.users.begin(); uit !=
             it->second.users.end(); ++uit)
          if (pre.find(*uit) == pre.end())
          {
#ifdef DEBUG_LEGION
            assert(rewrite.find(*uit) != rewrite.end());
#endif
            unsigned frontier = rewrite[*uit];
            if (frontiers.find(frontier) == frontiers.end())
            {
              unsigned next_event_id = events.size();
              frontiers[frontier] = next_event_id;
              events.resize(next_event_id + 1);
            }
            new_last_users[it->first].users.insert(frontier);
          }
      }
      last_users.swap(new_last_users);

      new_instructions.clear();
      unsigned next_instruction_id = instructions.size();
      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
        Instruction *inst = instructions[idx];
        MergeEvent *new_merge = NULL;
        std::set<unsigned> users;

        switch (inst->get_kind())
        {
          case ISSUE_COPY:
            {
              IssueCopy *copy = inst->as_issue_copy();
              if (copy->precondition_idx == fence_completion_id)
              {
                for (unsigned idx = 0; idx < copy->src_fields.size(); ++idx)
                {
                  const CopySrcDstField &field = copy->src_fields[idx];
                  find_last_users(field.inst, field.field_id, users);
                }
                for (unsigned idx = 0; idx < copy->dst_fields.size(); ++idx)
                {
                  const CopySrcDstField &field = copy->dst_fields[idx];
                  find_last_users(field.inst, field.field_id, users);
                }
                if (users.size() == 1)
                  copy->precondition_idx = *users.begin();
                else
                {
                  new_merge = new MergeEvent(*this, next_instruction_id, users);
                  copy->precondition_idx = next_instruction_id;
                  ++next_instruction_id;
                }
              }
              break;
            }
          case ISSUE_FILL:
            {
              IssueFill *fill = inst->as_issue_fill();
              if (fill->precondition_idx == fence_completion_id)
              {
                for (unsigned idx = 0; idx < fill->fields.size(); ++idx)
                {
                  const CopySrcDstField &field = fill->fields[idx];
                  find_last_users(field.inst, field.field_id, users);
                }
                if (users.size() == 1)
                  fill->precondition_idx = *users.begin();
                else
                {
                  new_merge = new MergeEvent(*this, next_instruction_id, users);
                  fill->precondition_idx = next_instruction_id;
                  ++next_instruction_id;
                }
              }
              break;
            }
          case SET_READY_EVENT:
            {
              SetReadyEvent *ready = inst->as_set_ready_event();
              if (ready->ready_event_idx == fence_completion_id)
              {
                std::vector<FieldID> field_ids;
                InstanceView *view = ready->view;
                view->logical_node->get_column_source()
                  ->get_field_set(ready->fields, field_ids);
                const PhysicalInstance &inst =
                  view->get_manager()->get_instance();
                for (std::vector<FieldID>::iterator it = field_ids.begin();
                    it != field_ids.end(); ++it)
                  find_last_users(inst, *it, users);
                if (users.size() == 1)
                  ready->ready_event_idx = *users.begin();
                else
                {
                  new_merge = new MergeEvent(*this, next_instruction_id, users);
                  ready->ready_event_idx = next_instruction_id;
                  ++next_instruction_id;
                }
              }
              break;
            }
          default:
            {
              break;
            }
        }

        if (new_merge != NULL)
          new_instructions.push_back(new_merge);
        new_instructions.push_back(inst);
      }
      instructions.swap(new_instructions);

      for (std::map<TraceLocalID, Operation*>::iterator it =
           operations.begin(); it != operations.end(); ++it)
        if (it->second->get_operation_kind() == Operation::TASK_OP_KIND)
          instructions.push_back(new LaunchTask(*this, it->first));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::schedule_instructions(void)
    //--------------------------------------------------------------------------
    {
      std::vector<TraceLocalID> inst_owners;
      std::vector<unsigned> gen(events.size(), -1U);
      inst_owners.resize(instructions.size());

      for (unsigned idx = 1; idx < instructions.size(); ++idx)
      {
        inst_owners[idx].first = -1U;
        Instruction *inst = instructions[idx];
        switch (inst->get_kind())
        {
          case GET_TERM_EVENT:
          case GET_COPY_TERM_EVENT:
          case CREATE_AP_USER_EVENT:
          case MERGE_EVENT:
          case ISSUE_COPY:
          case ISSUE_FILL:
          case SET_COPY_SYNC_EVENT:
            {
              gen[inst->get_lhs()] = idx;
              break;
            }
          default:
            {
              break;
            }
        }
      }

      for (unsigned idx = 1; idx < instructions.size(); ++idx)
      {
        Instruction *inst = instructions[idx];
        switch (inst->get_kind())
        {
          case ISSUE_COPY:
            {
              IssueCopy *copy = inst->as_issue_copy();
#ifdef DEBUG_LEGION
              assert(copy != NULL);
#endif
              TraceLocalID owner = copy->op_key;
              inst_owners[idx] = owner;
              unsigned precondition_gen_idx = gen[copy->precondition_idx];
              if (precondition_gen_idx != -1U &&
                  inst_owners[precondition_gen_idx].first == -1U)
                inst_owners[precondition_gen_idx] = owner;
              break;
            }
          case ISSUE_FILL:
            {
              IssueFill *fill = inst->as_issue_fill();
#ifdef DEBUG_LEGION
              assert(fill != NULL);
#endif
              TraceLocalID owner = fill->op_key;
              inst_owners[idx] = owner;
              unsigned precondition_gen_idx = gen[fill->precondition_idx];
              if (precondition_gen_idx != -1U &&
                  inst_owners[precondition_gen_idx].first == -1U)
                inst_owners[precondition_gen_idx] = owner;
              break;
            }
          case SET_READY_EVENT:
            {
              SetReadyEvent *set_ready = inst->as_set_ready_event();
#ifdef DEBUG_LEGION
              assert(set_ready != NULL);
#endif
              TraceLocalID owner = set_ready->op_key;
              inst_owners[idx] = owner;
              unsigned ready_event_gen_idx = gen[set_ready->ready_event_idx];
              if (ready_event_gen_idx != -1U &&
                  inst_owners[ready_event_gen_idx].first == -1U)
                inst_owners[ready_event_gen_idx] = owner;
              break;
            }
          case TRIGGER_EVENT:
            {
              TriggerEvent *trigger_event = inst->as_trigger_event();
              unsigned lhs_gen_idx = gen[trigger_event->lhs];
              unsigned rhs_gen_idx = gen[trigger_event->rhs];
#ifdef DEBUG_LEGION
              assert(lhs_gen_idx != -1U);
              assert(rhs_gen_idx != -1U);
              assert(inst_owners[rhs_gen_idx].first != -1U);
#endif
              inst_owners[idx] = inst_owners[rhs_gen_idx];
              inst_owners[lhs_gen_idx] = inst_owners[rhs_gen_idx];
              break;
            }
          case TRIGGER_COPY_COMPLETION:
            {
              TriggerCopyCompletion *trigger_complete =
                inst->as_trigger_copy_completion();
              unsigned rhs_gen_idx = gen[trigger_complete->rhs];
#ifdef DEBUG_LEGION
              assert(rhs_gen_idx != -1U);
#endif
              TraceLocalID owner = trigger_complete->lhs;
              inst_owners[idx] = owner;
              if (inst_owners[rhs_gen_idx].first == -1U)
                inst_owners[rhs_gen_idx] = owner;
              break;
            }
          case GET_TERM_EVENT:
          case GET_COPY_TERM_EVENT:
          case SET_COPY_SYNC_EVENT:
          case LAUNCH_TASK:
            {
              inst_owners[idx] = inst->get_owner();
              break;
            }
          case CREATE_AP_USER_EVENT:
          case MERGE_EVENT:
            {
              break;
            }
          default:
            {
              assert(false);
              break;
            }
        }
      }

      bool done = false;
      while (!done)
      {
        done = true;
        for (unsigned idx = instructions.size() - 1; idx > 0; --idx)
        {
          Instruction *inst = instructions[idx];
          if (inst->get_kind() == MERGE_EVENT)
          {
            MergeEvent *merge = inst->as_merge_event();
            if (inst_owners[idx].first != -1U)
            {
              const TraceLocalID &owner = inst_owners[idx];
              const std::set<unsigned> &rhs = merge->rhs;
              for (std::set<unsigned>::iterator it = rhs.begin();
                   it != rhs.end(); ++it)
              {
                unsigned rh_gen = gen[*it];
                if (rh_gen != -1U && inst_owners[rh_gen].first == -1U)
                  inst_owners[rh_gen] = owner;
              }
            }
            else
              done = false;
          }
        }
      }

      for (unsigned idx = 1; idx < instructions.size(); ++idx)
      {
        const TraceLocalID &owner = inst_owners[idx];
#ifdef DEBUG_LEGION
        assert(owner.first != -1U);
#endif
        schedules[owner].push_back(instructions[idx]);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ inline std::string PhysicalTemplate::view_to_string(
                                                       const InstanceView *view)
    //--------------------------------------------------------------------------
    {
      assert(view->logical_node->is_region());
      std::stringstream ss;
      LogicalRegion handle = view->logical_node->as_region_node()->handle;
      ss << "pointer: " << std::hex << view
         << ", instance: " << std::hex << view->get_manager()->get_instance().id
         << ", kind: "<< (view->is_materialized_view() ? "   normal" : "reduction")
         << ", region: " << "(" << handle.get_index_space().get_id()
         << "," << handle.get_field_space().get_id()
         << "," << handle.get_tree_id()
         << ")";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    /*static*/ inline std::string PhysicalTemplate::view_to_string(
                                                       const FillView *view)
    //--------------------------------------------------------------------------
    {
      assert(view->logical_node->is_region());
      std::stringstream ss;
      LogicalRegion handle = view->logical_node->as_region_node()->handle;
      ss << "pointer: " << std::hex << view
         << ", region: " << "(" << handle.get_index_space().get_id()
         << "," << handle.get_field_space().get_id()
         << "," << handle.get_tree_id()
         << ")";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::dump_template(void)
    //--------------------------------------------------------------------------
    {
      std::cerr << "#### " << (replayable ? "Replayable" : "Non-replayable")
                << " Template " << this << " ####" << std::endl;
      std::cerr << "[Instructions]" << std::endl;
      if (schedules.size() > 0)
        for (std::map<TraceLocalID,std::vector<Instruction*> >::iterator it =
             schedules.begin(); it != schedules.end(); ++it)
        {
          const TraceLocalID &op_key = it->first;
          std::stringstream ss;
          ss << "operations[" << op_key.first << ",";
          if (op_key.second.dim > 1) ss << "(";
          for (int dim = 0; dim < op_key.second.dim; ++dim)
          {
            if (dim > 0) ss << ",";
            ss << op_key.second[dim];
          }
          if (op_key.second.dim > 1) ss << ")";
          ss << ")]";
          std::cerr << "  " << ss.str() << std::endl;
          for (std::vector<Instruction*>::iterator iit = it->second.begin();
               iit != it->second.end(); ++iit)
          std::cerr << "    " << (*iit)->to_string() << std::endl;
        }
      else
        for (std::vector<Instruction*>::iterator it = instructions.begin();
            it != instructions.end(); ++it)
          std::cerr << "  " << (*it)->to_string() << std::endl;
      for (std::map<unsigned, unsigned>::iterator it = frontiers.begin();
           it != frontiers.end(); ++it)
        std::cerr << "  events[" << it->second << "] = events["
                  << it->first << "]" << std::endl;
      std::cerr << "[Previous Valid Views]" << std::endl;
      for (LegionMap<InstanceView*, FieldMask>::aligned::iterator it =
           previous_valid_views.begin(); it !=
           previous_valid_views.end(); ++it)
      {
        char *mask = it->second.to_string();
        std::cerr << "  " << view_to_string(it->first) << " " << mask
                  << " logical ctx: " << logical_contexts[it->first]
                  << " physical ctx: " << physical_contexts[it->first];
        if (it->first->is_reduction_view())
          std::cerr << " initialized: " << initialized[it->first];
        std::cerr << std::endl;
        free(mask);
      }

      std::cerr << "[Previous Fill Views]" << std::endl;
      for (LegionMap<FillView*, FieldMask>::aligned::iterator it =
           untracked_fill_views.begin(); it != untracked_fill_views.end(); ++it)
      {
        char *mask = it->second.to_string();
        std::cerr << "  " << view_to_string(it->first) << " " << mask
                  << " " << std::endl;
        free(mask);
      }

      std::cerr << "[Valid Views]" << std::endl;
      for (LegionMap<InstanceView*, FieldMask>::aligned::iterator it =
           valid_views.begin(); it != valid_views.end(); ++it)
      {
        char *mask = it->second.to_string();
        std::cerr << "  " << view_to_string(it->first) << " " << mask
                  << " logical ctx: " << logical_contexts[it->first]
                  << " physical ctx: " << physical_contexts[it->first]
                  << std::endl;
        free(mask);
      }

      std::cerr << "[Reduction Views]" << std::endl;
      for (LegionMap<InstanceView*, FieldMask>::aligned::iterator it =
           reduction_views.begin(); it != reduction_views.end(); ++it)
      {
        char *mask = it->second.to_string();
        std::cerr << "  " << view_to_string(it->first) << " " << mask
                  << " logical ctx: " << logical_contexts[it->first]
                  << " physical ctx: " << physical_contexts[it->first]
                  << " initialized: " << initialized[it->first] << std::endl;
        free(mask);
      }

      std::cerr << "[Fill Views]" << std::endl;
      for (LegionMap<FillView*, FieldMask>::aligned::iterator it =
           fill_views.begin(); it != fill_views.end(); ++it)
      {
        char *mask = it->second.to_string();
        std::cerr << "  " << view_to_string(it->first) << " " << mask
                  << " " << std::endl;
        free(mask);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::sanity_check(void)
    //--------------------------------------------------------------------------
    {
      // Reduction instances should not have been recycled in the trace
      for (LegionMap<InstanceView*, FieldMask>::aligned::iterator it =
           previous_valid_views.begin(); it !=
           previous_valid_views.end(); ++it)
        if (it->first->is_reduction_view())
          assert(initialized.find(it->first) != initialized.end() &&
                 initialized[it->first]);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_mapper_output(SingleTask *task,
                                            const Mapper::MapTaskOutput &output,
                              const std::deque<InstanceSet> &physical_instances)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(template_lock);

      TraceLocalID op_key = task->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(cached_mappings.find(op_key) == cached_mappings.end());
#endif
      CachedMapping &mapping = cached_mappings[op_key];
      mapping.target_procs = output.target_procs;
      mapping.chosen_variant = output.chosen_variant;
      mapping.task_priority = output.task_priority;
      mapping.postmap_task = output.postmap_task;
      mapping.physical_instances = physical_instances;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::get_mapper_output(SingleTask *task,
                                             VariantID &chosen_variant,
                                             TaskPriority &task_priority,
                                             bool &postmap_task,
                              std::vector<Processor> &target_procs,
                              std::deque<InstanceSet> &physical_instances) const
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(template_lock, 1, false/*exclusive*/);

      TraceLocalID op_key = task->get_trace_local_id();
      CachedMappings::const_iterator finder = cached_mappings.find(op_key);
#ifdef DEBUG_LEGION
      assert(finder != cached_mappings.end());
#endif
      chosen_variant = finder->second.chosen_variant;
      task_priority = finder->second.task_priority;
      postmap_task = finder->second.postmap_task;
      target_procs = finder->second.target_procs;
      physical_instances = finder->second.physical_instances;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_get_term_event(ApEvent lhs, SingleTask* task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(task->is_memoizing());
#endif
      AutoLock tpl_lock(template_lock);

      unsigned lhs_ = events.size();
      events.push_back(lhs);
#ifdef DEBUG_LEGION
      assert(event_map.find(lhs) == event_map.end());
#endif
      event_map[lhs] = lhs_;

      TraceLocalID key = task->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(operations.find(key) == operations.end());
      assert(task_entries.find(key) == task_entries.end());
#endif
      operations[key] = task;
      task_entries[key] = instructions.size();

      instructions.push_back(new GetTermEvent(*this, lhs_, key));
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_create_ap_user_event(ApUserEvent lhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs.exists());
#endif
      AutoLock tpl_lock(template_lock);

      unsigned lhs_ = events.size();
      user_events.resize(events.size());
      events.push_back(lhs);
      user_events.push_back(lhs);
#ifdef DEBUG_LEGION
      assert(event_map.find(lhs) == event_map.end());
#endif
      event_map[lhs] = lhs_;
      instructions.push_back(new CreateApUserEvent(*this, lhs_));
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_trigger_event(ApUserEvent lhs, ApEvent rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs.exists());
      assert(rhs.exists());
#endif
      AutoLock tpl_lock(template_lock);

      events.push_back(ApEvent());
      std::map<ApEvent, unsigned>::iterator lhs_finder = event_map.find(lhs);
      std::map<ApEvent, unsigned>::iterator rhs_finder = event_map.find(rhs);
#ifdef DEBUG_LEGION
      assert(lhs_finder != event_map.end());
      assert(rhs_finder != event_map.end());
#endif
      unsigned lhs_ = lhs_finder->second;
      unsigned rhs_ = rhs_finder->second;
      instructions.push_back(new TriggerEvent(*this, lhs_, rhs_));
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(ApEvent &lhs, ApEvent rhs_)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> rhs;
      rhs.insert(rhs_);
      record_merge_events(lhs, rhs);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(ApEvent &lhs, ApEvent e1,
                                               ApEvent e2)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> rhs;
      rhs.insert(e1);
      rhs.insert(e2);
      record_merge_events(lhs, rhs);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(ApEvent &lhs, ApEvent e1,
                                               ApEvent e2, ApEvent e3)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> rhs;
      rhs.insert(e1);
      rhs.insert(e2);
      rhs.insert(e3);
      record_merge_events(lhs, rhs);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(ApEvent &lhs,
                                               const std::set<ApEvent>& rhs)
    //--------------------------------------------------------------------------
    {
#ifndef LEGION_SPY
      if (!lhs.exists() || (rhs.find(lhs) != rhs.end()))
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        if (rhs.find(lhs) != rhs.end())
          rename.trigger(lhs);
        else
          rename.trigger();
        lhs = ApEvent(rename);
      }
#endif

      AutoLock tpl_lock(template_lock);

      unsigned lhs_ = events.size();
      events.push_back(lhs);
#ifdef DEBUG_LEGION
      assert(event_map.find(lhs) == event_map.end());
#endif
      event_map[lhs] = lhs_;

      std::set<unsigned> rhs_;
      rhs_.insert(fence_completion_id);
      for (std::set<ApEvent>::const_iterator it = rhs.begin(); it != rhs.end();
           it++)
      {
        std::map<ApEvent, unsigned>::iterator finder = event_map.find(*it);
        if (finder != event_map.end())
          rhs_.insert(finder->second);
      }
#ifdef DEBUG_LEGION
      assert(rhs_.size() > 0);
#endif

      instructions.push_back(new MergeEvent(*this, lhs_, rhs_));
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_copy_views(InstanceView *src,
                                             const FieldMask &src_mask,
                                             ContextID src_logical_ctx,
                                             ContextID src_physical_ctx,
                                             InstanceView *dst,
                                             const FieldMask &dst_mask,
                                             ContextID dst_logical_ctx,
                                             ContextID dst_physical_ctx)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(template_lock);

      if (src->is_reduction_view())
      {
        LegionMap<InstanceView*, FieldMask>::aligned::iterator finder =
          reduction_views.find(src);
        if (finder == reduction_views.end())
        {
          LegionMap<InstanceView*, FieldMask>::aligned::iterator pfinder =
            previous_valid_views.find(src);
          if (pfinder == previous_valid_views.end())
            previous_valid_views[src] = src_mask;
          else
            pfinder->second |= src_mask;
          initialized[src] = true;
        }
#ifdef DEBUG_LEGION
        else
        {
          assert(finder->second == src_mask);
          assert(initialized.find(src) != initialized.end());
        }
#endif
      }
      else
      {
        LegionMap<InstanceView*, FieldMask>::aligned::iterator finder =
          valid_views.find(src);
        if (finder == valid_views.end())
        {
          LegionMap<InstanceView*, FieldMask>::aligned::iterator pfinder =
            previous_valid_views.find(src);
          if (pfinder == previous_valid_views.end())
            previous_valid_views[src] = src_mask;
          else
            pfinder->second |= src_mask;
        }
        else
          finder->second |= src_mask;
      }

#ifdef DEBUG_LEGION
      assert(!dst->is_reduction_view());
#endif
      LegionMap<InstanceView*, FieldMask>::aligned::iterator finder =
        valid_views.find(dst);
      if (finder == valid_views.end())
        valid_views[dst] = dst_mask;
      else
        finder->second |= dst_mask;

#ifdef DEBUG_LEGION
      assert(logical_contexts.find(src) == logical_contexts.end() ||
             logical_contexts[src] == src_logical_ctx);
      assert(logical_contexts.find(dst) == logical_contexts.end() ||
             logical_contexts[dst] == dst_logical_ctx);
      assert(physical_contexts.find(src) == physical_contexts.end() ||
             physical_contexts[src] == src_physical_ctx);
      assert(physical_contexts.find(dst) == physical_contexts.end() ||
             physical_contexts[dst] == dst_physical_ctx);
#endif
      logical_contexts[src] = src_logical_ctx;
      logical_contexts[dst] = dst_logical_ctx;
      physical_contexts[src] = src_physical_ctx;
      physical_contexts[dst] = dst_physical_ctx;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_issue_copy(Operation* op, ApEvent &lhs,
                                             RegionNode *node,
                                 const std::vector<CopySrcDstField>& src_fields,
                                 const std::vector<CopySrcDstField>& dst_fields,
                                             ApEvent precondition,
                                             PredEvent predicate_guard,
                                             RegionTreeNode *intersect,
                                             ReductionOpID redop,
                                             bool reduction_fold)
    //--------------------------------------------------------------------------
    {
      if (!lhs.exists())
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        rename.trigger();
        lhs = ApEvent(rename);
      }
      AutoLock tpl_lock(template_lock);

      unsigned lhs_ = events.size();
      events.push_back(lhs);
#ifdef DEBUG_LEGION
      assert(event_map.find(lhs) == event_map.end());
#endif
      event_map[lhs] = lhs_;

      Memoizable *memoizable = op->get_memoizable();
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      TraceLocalID op_key = memoizable->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(operations.find(op_key) != operations.end());
#endif

      std::map<ApEvent, unsigned>::iterator pre_finder =
        event_map.find(precondition);
#ifdef DEBUG_LEGION
      assert(pre_finder != event_map.end());
#endif

      for (unsigned idx = 0; idx < src_fields.size(); ++idx)
      {
        const CopySrcDstField &field = src_fields[idx];
        record_last_user(field.inst, node, field.field_id, lhs_, true);
      }
      for (unsigned idx = 0; idx < dst_fields.size(); ++idx)
      {
        const CopySrcDstField &field = dst_fields[idx];
        record_last_user(field.inst, node, field.field_id, lhs_, false);
      }

      unsigned precondition_idx = pre_finder->second;
      instructions.push_back(new IssueCopy(
            *this, lhs_, node, op_key, src_fields, dst_fields,
            precondition_idx, predicate_guard,
            intersect, redop, reduction_fold));
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_empty_copy(CompositeView *src,
                                             const FieldMask &src_mask,
                                             MaterializedView *dst,
                                             const FieldMask &dst_mask,
                                             ContextID logical_ctx)
    //--------------------------------------------------------------------------
    {
      // FIXME: Nested composite views potentially make the check expensive.
      //        Here we simply handle the case we know can be done efficiently.
      if (!src->has_nested_views())
      {
        bool already_valid = false;
        LegionMap<InstanceView*, FieldMask>::aligned::iterator finder =
          valid_views.find(dst);
        if (finder == valid_views.end())
          valid_views[dst] = dst_mask;
        else
        {
          already_valid = !(dst_mask - finder->second);
          finder->second |= dst_mask;
        }
        if (already_valid) return;

        src->closed_tree->record_closed_tree(src_mask, logical_ctx,
            previous_open_nodes, previous_projections);
      }
    }

    //--------------------------------------------------------------------------
    inline void PhysicalTemplate::record_ready_view(const RegionRequirement &req,
                                                    InstanceView *view,
                                                    const FieldMask &fields,
                                                    ContextID logical_ctx,
                                                    ContextID physical_ctx)
    //--------------------------------------------------------------------------
    {
      if (view->is_reduction_view())
      {
#ifdef DEBUG_LEGION
        assert(IS_REDUCE(req));
        assert(reduction_views.find(view) == reduction_views.end());
        assert(initialized.find(view) == initialized.end());
#endif
        reduction_views[view] = fields;
        initialized[view] = false;
      }
      else
      {
        LegionMap<InstanceView*, FieldMask>::aligned::iterator finder =
          valid_views.find(view);
        if (finder == valid_views.end())
        {
          if (HAS_READ(req))
          {
            LegionMap<InstanceView*, FieldMask>::aligned::iterator pfinder =
              previous_valid_views.find(view);
            if (pfinder == previous_valid_views.end())
              previous_valid_views[view] = fields;
            else
              pfinder->second |= fields;
          }
          valid_views[view] = fields;
        }
        else
          finder->second |= fields;
      }
#ifdef DEBUG_LEGION
      assert(logical_contexts.find(view) == logical_contexts.end() ||
             logical_contexts[view] == logical_ctx);
      assert(physical_contexts.find(view) == physical_contexts.end() ||
             physical_contexts[view] == physical_ctx);
#endif
      logical_contexts[view] = logical_ctx;
      physical_contexts[view] = physical_ctx;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_set_ready_event(Operation *op,
                                                  unsigned region_idx,
                                                  unsigned inst_idx,
                                                  ApEvent ready_event,
                                                  const RegionRequirement &req,
                                                  InstanceView *view,
                                                  const FieldMask &fields,
                                                  ContextID logical_ctx,
                                                  ContextID physical_ctx)
    //--------------------------------------------------------------------------
    {
      if (op->get_operation_kind() == Operation::COPY_OP_KIND) return;

      AutoLock tpl_lock(template_lock);

      Memoizable *memoizable = op->get_memoizable();
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      TraceLocalID op_key = memoizable->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(operations.find(op_key) != operations.end());
#endif

      std::map<ApEvent, unsigned>::iterator ready_finder =
        event_map.find(ready_event);
#ifdef DEBUG_LEGION
      assert(ready_finder != event_map.end());
#endif
      unsigned ready_event_idx = ready_finder->second;

      RegionNode *region_node = view->logical_node->as_region_node();
      if (view->is_reduction_view())
      {
        ReductionView *reduction_view = view->as_reduction_view();
        PhysicalManager *manager = reduction_view->get_manager();
        LayoutDescription *const layout = manager->layout;
        const ReductionOp *reduction_op =
          Runtime::get_reduction_op(reduction_view->get_redop());

        std::vector<CopySrcDstField> fields;
        {
          std::vector<FieldID> fill_fields;
          layout->get_fields(fill_fields);
          layout->compute_copy_offsets(fill_fields, manager, fields);
        }

        unsigned lhs_ = events.size();
        events.push_back(ApEvent());

        void *fill_buffer = malloc(reduction_op->sizeof_rhs);
        reduction_op->init(fill_buffer, 1);
#ifdef DEBUG_LEGION
        assert(view->logical_node->is_region());
#endif
        instructions.push_back(
            new IssueFill(*this, lhs_, region_node,
                          op_key, fields, fill_buffer, reduction_op->sizeof_rhs,
                          ready_event_idx, PredEvent::NO_PRED_EVENT,
#ifdef LEGION_SPY
                          0,
#endif
                          NULL));
        for (unsigned idx = 0; idx < fields.size(); ++idx)
        {
          const CopySrcDstField &field = fields[idx];
          record_last_user(field.inst, region_node, field.field_id, lhs_, true);
        }
        free(fill_buffer);
        ready_event_idx = lhs_;
      }

      events.push_back(ApEvent());
      instructions.push_back(
          new SetReadyEvent(*this, op_key, region_idx, inst_idx,
                            ready_event_idx, view, fields));
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif

      record_ready_view(req, view, fields, logical_ctx, physical_ctx);

      std::map<TraceLocalID, unsigned>::iterator finder =
        task_entries.find(op_key);
#ifdef DEBUG_LEGION
      assert(finder != task_entries.end());
#endif
      std::vector<FieldID> field_ids;
      region_node->get_column_source()->get_field_set(fields, field_ids);
      const PhysicalInstance &inst = view->get_manager()->get_instance();
      for (std::vector<FieldID>::iterator it = field_ids.begin(); it !=
           field_ids.end(); ++it)
        record_last_user(inst, region_node, *it, finder->second,
                         IS_READ_ONLY(req));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_get_copy_term_event(ApEvent lhs, CopyOp* copy)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(copy->is_memoizing());
#endif
      AutoLock tpl_lock(template_lock);

      unsigned lhs_ = events.size();
      events.push_back(lhs);
#ifdef DEBUG_LEGION
      assert(event_map.find(lhs) == event_map.end());
#endif
      event_map[lhs] = lhs_;

      TraceLocalID key = copy->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(operations.find(key) == operations.end());
      assert(task_entries.find(key) == task_entries.end());
#endif
      operations[key] = copy;
      task_entries[key] = instructions.size();

      instructions.push_back(new GetCopyTermEvent(*this, lhs_, key));
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_set_copy_sync_event(
                                                     ApEvent &lhs, CopyOp* copy)
    //--------------------------------------------------------------------------
    {
      if (!lhs.exists())
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        rename.trigger();
        lhs = ApEvent(rename);
      }
#ifdef DEBUG_LEGION
      assert(copy->is_memoizing());
#endif
      AutoLock tpl_lock(template_lock);

      unsigned lhs_ = events.size();
      events.push_back(lhs);
#ifdef DEBUG_LEGION
      assert(event_map.find(lhs) == event_map.end());
#endif
      event_map[lhs] = lhs_;

      TraceLocalID key = copy->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(operations.find(key) != operations.end());
      assert(task_entries.find(key) != task_entries.end());
#endif
      instructions.push_back(new SetCopySyncEvent(*this, lhs_, key));
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_trigger_copy_completion(
                                                      CopyOp* copy, ApEvent rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(copy->is_memoizing());
#endif
      AutoLock tpl_lock(template_lock);

      events.push_back(ApEvent());
      TraceLocalID lhs_ = copy->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(event_map.find(rhs) != event_map.end());
#endif
      unsigned rhs_ = event_map[rhs];

#ifdef DEBUG_LEGION
      assert(operations.find(lhs_) != operations.end());
      assert(task_entries.find(lhs_) != task_entries.end());
#endif
      instructions.push_back(new TriggerCopyCompletion(*this, lhs_, rhs_));

#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_issue_fill(Operation *op, ApEvent &lhs,
                                             RegionNode *node,
                                     const std::vector<CopySrcDstField> &fields,
                                             const void *fill_buffer,
                                             size_t fill_size,
                                             ApEvent precondition,
                                             PredEvent predicate_guard,
#ifdef LEGION_SPY
                                             UniqueID fill_uid,
#endif
                                             RegionTreeNode *intersect)
    //--------------------------------------------------------------------------
    {
      if (!lhs.exists())
      {
        Realm::UserEvent rename(Realm::UserEvent::create_user_event());
        rename.trigger();
        lhs = ApEvent(rename);
      }
#ifdef DEBUG_LEGION
      assert(op->is_memoizing());
#endif
      AutoLock tpl_lock(template_lock);

      unsigned lhs_ = events.size();
      events.push_back(lhs);
#ifdef DEBUG_LEGION
      assert(event_map.find(lhs) == event_map.end());
#endif
      event_map[lhs] = lhs_;

      Memoizable *memoizable = op->get_memoizable();
#ifdef DEBUG_LEGION
      assert(memoizable != NULL);
#endif
      TraceLocalID key = memoizable->get_trace_local_id();
#ifdef DEBUG_LEGION
      assert(operations.find(key) != operations.end());
      assert(task_entries.find(key) != task_entries.end());
#endif

      std::map<ApEvent, unsigned>::iterator pre_finder =
        event_map.find(precondition);
#ifdef DEBUG_LEGION
      assert(pre_finder != event_map.end());
#endif
      unsigned precondition_idx = pre_finder->second;

      instructions.push_back(new IssueFill(*this, lhs_, node, key,
                             fields, fill_buffer, fill_size, precondition_idx,
                             predicate_guard,
#ifdef LEGION_SPY
                             fill_uid,
#endif
                             intersect));
#ifdef DEBUG_LEGION
      assert(instructions.size() == events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_fill_view(
                                FillView *fill_view, const FieldMask &fill_mask)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
#ifdef DEBUG_LEGION
      assert(fill_views.find(fill_view) == fill_views.end());
#endif
      fill_views[fill_view] = fill_mask;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_deferred_copy_from_fill_view(
                                                            FillView *fill_view,
                                                         InstanceView* dst_view,
                                                     const FieldMask &copy_mask,
                                                          ContextID logical_ctx,
                                                         ContextID physical_ctx)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      LegionMap<FillView*, FieldMask>::aligned::iterator finder =
        fill_views.find(fill_view);
      if (finder == fill_views.end())
      {
        finder = untracked_fill_views.find(fill_view);
        if (finder == untracked_fill_views.end())
          untracked_fill_views[fill_view] = copy_mask;
        else
          finder->second |= copy_mask;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!(copy_mask - finder->second));
#endif
        LegionMap<InstanceView*, FieldMask>::aligned::iterator finder =
          valid_views.find(dst_view);
        if (finder == valid_views.end())
          valid_views[dst_view] = copy_mask;
        else
          finder->second |= copy_mask;

#ifdef DEBUG_LEGION
        assert(logical_contexts.find(dst_view) == logical_contexts.end() ||
               logical_contexts[dst_view] == logical_ctx);
        assert(physical_contexts.find(dst_view) == physical_contexts.end() ||
               physical_contexts[dst_view] == physical_ctx);
#endif
        logical_contexts[dst_view] = logical_ctx;
        physical_contexts[dst_view] = physical_ctx;
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_empty_copy_from_fill_view(
                                                         InstanceView* dst_view,
                                                     const FieldMask &copy_mask,
                                                          ContextID logical_ctx,
                                                         ContextID physical_ctx)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      LegionMap<InstanceView*, FieldMask>::aligned::iterator finder =
        valid_views.find(dst_view);
      if (finder == valid_views.end())
        valid_views[dst_view] = copy_mask;
      else
        finder->second |= copy_mask;

#ifdef DEBUG_LEGION
      assert(logical_contexts.find(dst_view) == logical_contexts.end() ||
          logical_contexts[dst_view] == logical_ctx);
      assert(physical_contexts.find(dst_view) == physical_contexts.end() ||
          physical_contexts[dst_view] == physical_ctx);
#endif
      logical_contexts[dst_view] = logical_ctx;
      physical_contexts[dst_view] = physical_ctx;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_blocking_call(void)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      has_block = true;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_outstanding_gc_event(
                                         InstanceView *view, ApEvent term_event)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      outstanding_gc_events[view].insert(term_event);
    }

    //--------------------------------------------------------------------------
    inline void PhysicalTemplate::record_last_user(const PhysicalInstance &inst,
                                                   RegionNode *node,
                                                   unsigned field,
                                                   unsigned user, bool read)
    //--------------------------------------------------------------------------
    {
      InstanceAccess key(inst, field);
      std::map<InstanceAccess, UserInfo>::iterator finder =
        last_users.find(key);
      if (finder == last_users.end())
      {
        UserInfo &info = last_users[key];
        info.users.insert(user);
        info.nodes.insert(node);
        info.read = read;
      }
      else
      {
        if (finder->second.read == read)
        {
          if (read)
          {
            finder->second.users.insert(user);
            finder->second.nodes.insert(node);
          }
          else if (!finder->second.read && !read)
          {
            std::set<RegionNode*> &nodes = finder->second.nodes;

            bool interfere_with_any = false;
            for (std::set<RegionNode*>::iterator it = nodes.begin();
                 it != nodes.end(); ++it)
              if ((*it)->intersects_with(node, false))
              {
                interfere_with_any = true;
                break;
              }
            if (interfere_with_any)
            {
              finder->second.users.clear();
              finder->second.nodes.clear();
            }
            finder->second.users.insert(user);
            finder->second.nodes.insert(node);
          }
        }
        else
        {
          finder->second.users.clear();
          finder->second.users.insert(user);
          finder->second.nodes.clear();
          finder->second.nodes.insert(node);
          finder->second.read = read;
        }
      }
    }

    //--------------------------------------------------------------------------
    inline void PhysicalTemplate::find_last_users(const PhysicalInstance &inst,
                                                  unsigned field,
                                                  std::set<unsigned> &users)
    //--------------------------------------------------------------------------
    {
      InstanceAccess key(inst, field);
      std::map<InstanceAccess, UserInfo>::iterator finder =
        last_users.find(key);
#ifdef DEBUG_LEGION
      assert(finder != last_users.end());
#endif
      for (std::set<unsigned>::iterator it = finder->second.users.begin(); it !=
           finder->second.users.end(); ++it)
      {
#ifdef DEBUG_LEGION
        assert(frontiers.find(*it) != frontiers.end());
#endif
        users.insert(frontiers[*it]);
      }
    }

    /////////////////////////////////////////////////////////////
    // Instruction
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Instruction::Instruction(PhysicalTemplate& tpl)
      : operations(tpl.operations), events(tpl.events),
        user_events(tpl.user_events)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // GetTermEvent
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    GetTermEvent::GetTermEvent(PhysicalTemplate& tpl, unsigned l,
                               const TraceLocalID& r)
      : Instruction(tpl), lhs(l), rhs(r)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < events.size());
      assert(operations.find(rhs) != operations.end());
#endif
    }

    //--------------------------------------------------------------------------
    void GetTermEvent::execute(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(rhs) != operations.end());
      assert(operations.find(rhs)->second != NULL);

      SingleTask *task = dynamic_cast<SingleTask*>(operations[rhs]);
      assert(task != NULL);
#else
      SingleTask *task = static_cast<SingleTask*>(operations[rhs]);
#endif
      ApEvent completion_event = task->get_task_completion();
      events[lhs] = completion_event;
      task->replay_map_task_output();
    }

    //--------------------------------------------------------------------------
    std::string GetTermEvent::to_string(void)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = operations[(" << rhs.first << ",";
      if (rhs.second.dim > 1) ss << "(";
      for (int dim = 0; dim < rhs.second.dim; ++dim)
      {
        if (dim > 0) ss << ",";
        ss << rhs.second[dim];
      }
      if (rhs.second.dim > 1) ss << ")";
      ss << ")].get_task_termination()";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    Instruction* GetTermEvent::clone(PhysicalTemplate& tpl,
                                  const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator finder = rewrite.find(lhs);
#ifdef DEBUG_LEGION
      assert(finder != rewrite.end());
#endif
      return new GetTermEvent(tpl, finder->second, rhs);
    }

    /////////////////////////////////////////////////////////////
    // CreateApUserEvent
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CreateApUserEvent::CreateApUserEvent(PhysicalTemplate& tpl, unsigned l)
      : Instruction(tpl), lhs(l)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < events.size());
      assert(lhs < user_events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void CreateApUserEvent::execute(void)
    //--------------------------------------------------------------------------
    {
      ApUserEvent ev = Runtime::create_ap_user_event();
      events[lhs] = ev;
      user_events[lhs] = ev;
    }

    //--------------------------------------------------------------------------
    std::string CreateApUserEvent::to_string(void)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = Runtime::create_ap_user_event()";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    Instruction* CreateApUserEvent::clone(PhysicalTemplate& tpl,
                                    const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator lhs_finder =
        rewrite.find(lhs);
#ifdef DEBUG_LEGION
      assert(lhs_finder != rewrite.end());
#endif
      return new CreateApUserEvent(tpl, lhs_finder->second);
    }

    /////////////////////////////////////////////////////////////
    // TriggerEvent
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TriggerEvent::TriggerEvent(PhysicalTemplate& tpl, unsigned l, unsigned r)
      : Instruction(tpl), lhs(l), rhs(r)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < events.size());
      assert(lhs < user_events.size());
      assert(rhs < events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void TriggerEvent::execute(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(events[lhs].exists());
      assert(user_events[lhs].exists());
      assert(events[rhs].exists());
      assert(events[lhs].id == user_events[lhs].id);
#endif
      Runtime::trigger_event(user_events[lhs], events[rhs]);
    }

    //--------------------------------------------------------------------------
    std::string TriggerEvent::to_string(void)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "Runtime::trigger_event(events[" << lhs
         << "], events[" << rhs << "])";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    Instruction* TriggerEvent::clone(PhysicalTemplate& tpl,
                                    const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator lhs_finder =
        rewrite.find(lhs);
      std::map<unsigned, unsigned>::const_iterator rhs_finder =
        rewrite.find(rhs);
#ifdef DEBUG_LEGION
      assert(lhs_finder != rewrite.end());
      assert(rhs_finder != rewrite.end());
#endif
      return new TriggerEvent(tpl, lhs_finder->second, rhs_finder->second);
    }

    /////////////////////////////////////////////////////////////
    // MergeEvent
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MergeEvent::MergeEvent(PhysicalTemplate& tpl, unsigned l,
                           const std::set<unsigned>& r)
      : Instruction(tpl), lhs(l), rhs(r)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < events.size());
      assert(rhs.size() > 0);
      for (std::set<unsigned>::iterator it = rhs.begin(); it != rhs.end();
           ++it)
        assert(*it < events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void MergeEvent::execute(void)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> to_merge;
      for (std::set<unsigned>::iterator it = rhs.begin(); it != rhs.end();
           ++it)
      {
#ifdef DEBUG_LEGION
        assert(*it < events.size());
#endif
        to_merge.insert(events[*it]);
      }
      ApEvent result = Runtime::merge_events(to_merge);
      events[lhs] = result;
    }

    //--------------------------------------------------------------------------
    std::string MergeEvent::to_string(void)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = Runtime::merge_events(";
      unsigned count = 0;
      for (std::set<unsigned>::iterator it = rhs.begin(); it != rhs.end();
           ++it)
      {
        if (count++ != 0) ss << ",";
        ss << "events[" << *it << "]";
      }
      ss << ")";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    Instruction* MergeEvent::clone(PhysicalTemplate& tpl,
                                   const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return NULL;
    }

    /////////////////////////////////////////////////////////////
    // AssignFenceCompletion
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AssignFenceCompletion::AssignFenceCompletion(
                                              PhysicalTemplate& tpl, unsigned l)
      : Instruction(tpl), fence_completion(tpl.fence_completion), lhs(l)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void AssignFenceCompletion::execute(void)
    //--------------------------------------------------------------------------
    {
      events[lhs] = fence_completion;
    }

    //--------------------------------------------------------------------------
    std::string AssignFenceCompletion::to_string(void)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = fence_completion";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    Instruction* AssignFenceCompletion::clone(PhysicalTemplate& tpl,
                                  const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator finder = rewrite.find(lhs);
#ifdef DEBUG_LEGION
      assert(finder != rewrite.end());
#endif
      return new AssignFenceCompletion(tpl, finder->second);
    }

    /////////////////////////////////////////////////////////////
    // IssueCopy
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IssueCopy::IssueCopy(PhysicalTemplate& tpl,
                         unsigned l, RegionNode *n,
                         const TraceLocalID& key,
                         const std::vector<CopySrcDstField>& s,
                         const std::vector<CopySrcDstField>& d,
                         unsigned pi, PredEvent pg, RegionTreeNode *i,
                         ReductionOpID ro, bool rf)
      : Instruction(tpl), lhs(l), node(n), op_key(key), src_fields(s),
        dst_fields(d), precondition_idx(pi), predicate_guard(pg),
        intersect(i), redop(ro), reduction_fold(rf)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < events.size());
      assert(operations.find(op_key) != operations.end());
      assert(src_fields.size() > 0);
      assert(dst_fields.size() > 0);
      assert(precondition_idx < events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void IssueCopy::execute(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(op_key) != operations.end());
      assert(operations.find(op_key)->second != NULL);
#endif
      Operation *op = operations[op_key];
      ApEvent precondition = events[precondition_idx];
      PhysicalTraceInfo trace_info;
      events[lhs] = node->issue_copy(op, src_fields, dst_fields, precondition,
          predicate_guard, trace_info, intersect, redop, reduction_fold);
    }

    //--------------------------------------------------------------------------
    std::string IssueCopy::to_string(void)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = copy(operations[(" << op_key.first << ",";
      if (op_key.second.dim > 1) ss << "(";
      for (int dim = 0; dim < op_key.second.dim; ++dim)
      {
        if (dim > 0) ss << ",";
        ss << op_key.second[dim];
      }
      if (op_key.second.dim > 1) ss << ")";
      ss << ")], {";
      for (unsigned idx = 0; idx < src_fields.size(); ++idx)
      {
        ss << "(" << std::hex << src_fields[idx].inst.id
           << "," << std::dec << src_fields[idx].subfield_offset
           << "," << src_fields[idx].size
           << "," << src_fields[idx].field_id
           << "," << src_fields[idx].serdez_id << ")";
        if (idx != src_fields.size() - 1) ss << ",";
      }
      ss << "}, {";
      for (unsigned idx = 0; idx < dst_fields.size(); ++idx)
      {
        ss << "(" << std::hex << dst_fields[idx].inst.id
           << "," << std::dec << dst_fields[idx].subfield_offset
           << "," << dst_fields[idx].size
           << "," << dst_fields[idx].field_id
           << "," << dst_fields[idx].serdez_id << ")";
        if (idx != dst_fields.size() - 1) ss << ",";
      }
      ss << "}, events[" << precondition_idx << "]";

      if (redop != 0) ss << ", " << redop;
      ss << ")";

      return ss.str();
    }

    //--------------------------------------------------------------------------
    Instruction* IssueCopy::clone(PhysicalTemplate& tpl,
                                  const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator lfinder = rewrite.find(lhs);
      std::map<unsigned, unsigned>::const_iterator pfinder =
        rewrite.find(precondition_idx);
#ifdef DEBUG_LEGION
      assert(lfinder != rewrite.end());
      assert(pfinder != rewrite.end());
#endif
      return new IssueCopy(tpl, lfinder->second, node, op_key, src_fields,
        dst_fields, pfinder->second, predicate_guard, intersect, redop,
        reduction_fold);
    }

    /////////////////////////////////////////////////////////////
    // IssueFill
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IssueFill::IssueFill(PhysicalTemplate& tpl, unsigned l, RegionNode *n,
                         const TraceLocalID &key,
                         const std::vector<CopySrcDstField> &f,
                         const void *fb, size_t fs, unsigned pi,
                         PredEvent pg,
#ifdef LEGION_SPY
                         UniqueID u,
#endif
                         RegionTreeNode *i)
      : Instruction(tpl), lhs(l), node(n), op_key(key), fields(f),
        precondition_idx(pi), predicate_guard(pg),
#ifdef LEGION_SPY
        fill_uid(u),
#endif
        intersect(i)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < events.size());
      assert(operations.find(op_key) != operations.end());
      assert(fields.size() > 0);
      assert(precondition_idx < events.size());
#endif
      fill_size = fs;
      fill_buffer = malloc(fs);
      memcpy(fill_buffer, fb, fs);
    }

    //--------------------------------------------------------------------------
    IssueFill::~IssueFill(void)
    //--------------------------------------------------------------------------
    {
      free(fill_buffer);
    }

    //--------------------------------------------------------------------------
    void IssueFill::execute(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(op_key) != operations.end());
      assert(operations.find(op_key)->second != NULL);
#endif
      Operation *op = operations[op_key];
      ApEvent precondition = events[precondition_idx];

      PhysicalTraceInfo trace_info;
      events[lhs] = node->issue_fill(op, fields, fill_buffer,
          fill_size, precondition, predicate_guard,
#ifdef LEGION_SPY
          fill_uid,
#endif
          trace_info, intersect);
    }

    //--------------------------------------------------------------------------
    std::string IssueFill::to_string(void)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = fill({";
      for (unsigned idx = 0; idx < fields.size(); ++idx)
      {
        ss << "(" << std::hex << fields[idx].inst.id
           << "," << std::dec << fields[idx].subfield_offset
           << "," << fields[idx].size
           << "," << fields[idx].field_id
           << "," << fields[idx].serdez_id << ")";
        if (idx != fields.size() - 1) ss << ",";
      }
      ss << "}, events[" << precondition_idx << "])";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    Instruction* IssueFill::clone(PhysicalTemplate& tpl,
                                  const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator lfinder = rewrite.find(lhs);
      std::map<unsigned, unsigned>::const_iterator pfinder =
        rewrite.find(precondition_idx);
#ifdef DEBUG_LEGION
      assert(lfinder != rewrite.end());
      assert(pfinder != rewrite.end());
#endif
      return new IssueFill(tpl, lfinder->second, node, op_key, fields,
          fill_buffer, fill_size, pfinder->second, predicate_guard,
#ifdef LEGION_SPY
          fill_uid,
#endif
          intersect);
    }

    /////////////////////////////////////////////////////////////
    // SetReadyEvent
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SetReadyEvent::SetReadyEvent(PhysicalTemplate& tpl,
                                 const TraceLocalID& key, unsigned ri,
                                 unsigned ii, unsigned rei, InstanceView *v,
                                 const FieldMask &f)
      : Instruction(tpl), op_key(key), region_idx(ri), inst_idx(ii),
        ready_event_idx(rei), view(v), fields(f)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(op_key) != operations.end());
      assert(ready_event_idx < events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void SetReadyEvent::execute(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ready_event_idx < events.size());
      assert(operations.find(op_key) != operations.end());
      assert(operations.find(op_key)->second != NULL);

      SingleTask *task = dynamic_cast<SingleTask*>(operations[op_key]);
      assert(task != NULL);
#else
      SingleTask *task = static_cast<SingleTask*>(operations[op_key]);
#endif
      const std::deque<InstanceSet> &physical_instances =
        task->get_physical_instances();
      InstanceRef &ref =
        const_cast<InstanceRef&>(physical_instances[region_idx][inst_idx]);
      ApEvent ready_event = events[ready_event_idx];
      ref.set_ready_event(ready_event);
    }

    //--------------------------------------------------------------------------
    std::string SetReadyEvent::to_string(void)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "operations[(" << op_key.first << ",";
      if (op_key.second.dim > 1) ss << "(";
      for (int dim = 0; dim < op_key.second.dim; ++dim)
      {
        if (dim > 0) ss << ",";
        ss << op_key.second[dim];
      }
      if (op_key.second.dim > 1) ss << ")";
      ss << ")].get_physical_instances()["
         << region_idx << "]["
         << inst_idx << "].set_ready_event(events["
         << ready_event_idx << "])  (pointer: "
         << std::hex << view << ", instance id: "
         << std::hex << view->get_manager()->get_instance().id << ")";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    Instruction* SetReadyEvent::clone(PhysicalTemplate& tpl,
                                    const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator finder =
        rewrite.find(ready_event_idx);
#ifdef DEBUG_LEGION
      assert(finder != rewrite.end());
#endif
      return new SetReadyEvent(tpl, op_key, region_idx, inst_idx,
        finder->second, view, fields);
    }

    /////////////////////////////////////////////////////////////
    // GetCopyTermEvent
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    GetCopyTermEvent::GetCopyTermEvent(PhysicalTemplate& tpl, unsigned l,
                                       const TraceLocalID& r)
      : Instruction(tpl), lhs(l), rhs(r)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < events.size());
      assert(operations.find(rhs) != operations.end());
#endif
    }

    //--------------------------------------------------------------------------
    void GetCopyTermEvent::execute(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(rhs) != operations.end());
      assert(operations.find(rhs)->second != NULL);

      CopyOp *copy = dynamic_cast<CopyOp*>(operations[rhs]);
      assert(copy != NULL);
#else
      CopyOp *copy = static_cast<CopyOp*>(operations[rhs]);
#endif
      events[lhs] = copy->get_completion_event();
    }

    //--------------------------------------------------------------------------
    std::string GetCopyTermEvent::to_string(void)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = operations[(" << rhs.first << ",";
      if (rhs.second.dim > 1) ss << "(";
      for (int dim = 0; dim < rhs.second.dim; ++dim)
      {
        if (dim > 0) ss << ",";
        ss << rhs.second[dim];
      }
      if (rhs.second.dim > 1) ss << ")";
      ss << ")].get_completion_event()";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    Instruction* GetCopyTermEvent::clone(PhysicalTemplate& tpl,
                                    const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator finder = rewrite.find(lhs);
#ifdef DEBUG_LEGION
      assert(finder != rewrite.end());
#endif
      return new GetCopyTermEvent(tpl, finder->second, rhs);
    }

    /////////////////////////////////////////////////////////////
    // SetCopySyncEvent
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SetCopySyncEvent::SetCopySyncEvent(PhysicalTemplate& tpl, unsigned l,
                                       const TraceLocalID& r)
      : Instruction(tpl), lhs(l), rhs(r)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lhs < events.size());
      assert(operations.find(rhs) != operations.end());
#endif
    }

    //--------------------------------------------------------------------------
    void SetCopySyncEvent::execute(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(rhs) != operations.end());
      assert(operations.find(rhs)->second != NULL);

      CopyOp *copy = dynamic_cast<CopyOp*>(operations[rhs]);
      assert(copy != NULL);
#else
      CopyOp *copy = static_cast<CopyOp*>(operations[rhs]);
#endif
      ApEvent sync_condition = copy->compute_sync_precondition();
      events[lhs] = sync_condition;
    }

    //--------------------------------------------------------------------------
    std::string SetCopySyncEvent::to_string(void)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = operations[(" << rhs.first << ",";
      if (rhs.second.dim > 1) ss << "(";
      for (int dim = 0; dim < rhs.second.dim; ++dim)
      {
        if (dim > 0) ss << ",";
        ss << rhs.second[dim];
      }
      if (rhs.second.dim > 1) ss << ")";
      ss << ")].compute_sync_precondition()";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    Instruction* SetCopySyncEvent::clone(PhysicalTemplate& tpl,
                                    const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator finder = rewrite.find(lhs);
#ifdef DEBUG_LEGION
      assert(finder != rewrite.end());
#endif
      return new SetCopySyncEvent(tpl, finder->second, rhs);
    }

    /////////////////////////////////////////////////////////////
    // TriggerCopyCompletion
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TriggerCopyCompletion::TriggerCopyCompletion(PhysicalTemplate& tpl,
                                              const TraceLocalID& l, unsigned r)
      : Instruction(tpl), lhs(l), rhs(r)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(lhs) != operations.end());
      assert(rhs < events.size());
#endif
    }

    //--------------------------------------------------------------------------
    void TriggerCopyCompletion::execute(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(lhs) != operations.end());
      assert(operations.find(lhs)->second != NULL);

      CopyOp *copy = dynamic_cast<CopyOp*>(operations[lhs]);
      assert(copy != NULL);
#else
      CopyOp *copy = static_cast<CopyOp*>(operations[lhs]);
#endif
      copy->complete_copy_execution(events[rhs]);
      copy->remove_mapping_reference(copy->get_generation());
    }

    //--------------------------------------------------------------------------
    std::string TriggerCopyCompletion::to_string(void)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "operations[" << lhs.first << ",";
      if (lhs.second.dim > 1) ss << "(";
      for (int dim = 0; dim < lhs.second.dim; ++dim)
      {
        if (dim > 0) ss << ",";
        ss << lhs.second[dim];
      }
      if (lhs.second.dim > 1) ss << ")";
      ss << ")].complete_copy_execution(events[" << rhs << "])";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    Instruction* TriggerCopyCompletion::clone(PhysicalTemplate& tpl,
                                    const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned>::const_iterator finder = rewrite.find(rhs);
#ifdef DEBUG_LEGION
      assert(finder != rewrite.end());
#endif
      return new TriggerCopyCompletion(tpl, lhs, finder->second);
    }

    /////////////////////////////////////////////////////////////
    // LaunchTask
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LaunchTask::LaunchTask(PhysicalTemplate& tpl, const TraceLocalID& op)
      : Instruction(tpl), op_key(op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(op_key) != operations.end());
#endif
    }

    //--------------------------------------------------------------------------
    void LaunchTask::execute(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(operations.find(op_key) != operations.end());
      assert(operations.find(op_key)->second != NULL);

      SingleTask *task = dynamic_cast<SingleTask*>(operations[op_key]);
      assert(task != NULL);
#else
      SingleTask *task = static_cast<SingleTask*>(operations[op_key]);
#endif
      if (!task->arrive_barriers.empty())
      {
        ApEvent done_event = task->get_task_completion();
        for (std::vector<PhaseBarrier>::const_iterator it = 
             task->arrive_barriers.begin(); it != 
             task->arrive_barriers.end(); it++)
          Runtime::phase_barrier_arrive(*it, 1/*count*/, done_event);
      }
#ifdef DEBUG_LEGION
      assert(task->is_leaf() && !task->has_virtual_instances());
#endif
      task->launch_task();
      task->remove_mapping_reference(task->get_generation());
    }

    //--------------------------------------------------------------------------
    std::string LaunchTask::to_string(void)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "operations[" << op_key.first << ",";
      if (op_key.second.dim > 1) ss << "(";
      for (int dim = 0; dim < op_key.second.dim; ++dim)
      {
        if (dim > 0) ss << ",";
        ss << op_key.second[dim];
      }
      if (op_key.second.dim > 1) ss << ")";
      ss << ")].launch_task()";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    Instruction* LaunchTask::clone(PhysicalTemplate& tpl,
                                    const std::map<unsigned, unsigned> &rewrite)
    //--------------------------------------------------------------------------
    {
      return new LaunchTask(tpl, op_key);
    }

    /////////////////////////////////////////////////////////////
    // PhysicalTraceInfo
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalTraceInfo::PhysicalTraceInfo(void)
    //--------------------------------------------------------------------------
      : recording(false), op(NULL), tpl(NULL)
    {
    }

  }; // namespace Internal 
}; // namespace Legion


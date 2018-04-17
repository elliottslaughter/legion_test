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


#ifndef __LEGION_TRACE__
#define __LEGION_TRACE__

#include "legion.h"
#include "legion/runtime.h"
#include "legion/legion_ops.h"

namespace Legion {
  namespace Internal {

    /**
     * \class LegionTrace
     * This is the abstract base class for a trace object
     * and is used to support both static and dynamic traces
     */
    class LegionTrace : public Collectable {
    public:
      struct DependenceRecord {
      public:
        DependenceRecord(int idx)
          : operation_idx(idx), prev_idx(-1), next_idx(-1),
            validates(false), dtype(TRUE_DEPENDENCE) { }
        DependenceRecord(int op_idx, int pidx, int nidx,
                         bool val, DependenceType d,
                         const FieldMask &m)
          : operation_idx(op_idx), prev_idx(pidx), 
            next_idx(nidx), validates(val), dtype(d),
            dependent_mask(m) { }
      public:
        inline bool merge(const DependenceRecord &record)
        {
          if ((operation_idx != record.operation_idx) ||
              (prev_idx != record.prev_idx) ||
              (next_idx != record.next_idx) ||
              (validates != record.validates) ||
              (dtype != record.dtype))
            return false;
          dependent_mask |= record.dependent_mask;
          return true;
        }
      public:
        int operation_idx;
        int prev_idx; // previous region requirement index
        int next_idx; // next region requirement index
        bool validates;
        DependenceType dtype;
        FieldMask dependent_mask;
      };
      struct AliasChildren {
      public:
        AliasChildren(unsigned req_idx, unsigned dep, const FieldMask &m)
          : req_index(req_idx), depth(dep), mask(m) { }
      public:
        unsigned req_index;
        unsigned depth;
        FieldMask mask;
      };
    public:
      enum TracingState {
        LOGICAL_ONLY,
        PHYSICAL_RECORD,
        PHYSICAL_REPLAY,
      };
    public:
      LegionTrace(TaskContext *ctx, bool logical_only);
      virtual ~LegionTrace(void);
    public:
      virtual bool is_static_trace(void) const = 0;
      virtual bool is_dynamic_trace(void) const = 0;
      virtual StaticTrace* as_static_trace(void) = 0;
      virtual DynamicTrace* as_dynamic_trace(void) = 0;
      virtual TraceID get_trace_id(void) const = 0;
    public:
      virtual bool is_fixed(void) const = 0;
      virtual bool handles_region_tree(RegionTreeID tid) const = 0;
      virtual void record_static_dependences(Operation *op,
                     const std::vector<StaticDependence> *dependences) = 0;
      virtual void register_operation(Operation *op, GenerationID gen) = 0; 
      virtual void record_dependence(Operation *target, GenerationID target_gen,
                                Operation *source, GenerationID source_gen) = 0;
      virtual void record_region_dependence(
                                    Operation *target, GenerationID target_gen,
                                    Operation *source, GenerationID source_gen,
                                    unsigned target_idx, unsigned source_idx,
                                    DependenceType dtype, bool validates,
                                    const FieldMask &dependent_mask) = 0;
      virtual void record_aliased_children(unsigned req_index, unsigned depth,
                                           const FieldMask &aliased_mask) = 0;
    public:
      bool has_physical_trace(void) { return physical_trace != NULL; }
      PhysicalTrace* get_physical_trace(void) { return physical_trace; }
      void register_physical_only(Operation *op, GenerationID gen);
    public:
      void replay_aliased_children(std::vector<RegionTreePath> &paths) const;
      void end_trace_execution(FenceOp *fence_op);
    public:
      void initialize_tracing_state(void) { state = LOGICAL_ONLY; }
      void set_state_record(void) { state = PHYSICAL_RECORD; }
      void set_state_replay(void) { state = PHYSICAL_REPLAY; }
      bool is_recording(void) const { return state == PHYSICAL_RECORD; }
      bool is_replaying(void) const { return state == PHYSICAL_REPLAY; }
    public:
      void invalidate_trace_cache(void);
      void invalidate_current_template(void);
#ifdef LEGION_SPY
    public:
      UniqueID get_current_uid_by_index(unsigned op_idx) const;
#endif
    public:
      TaskContext *const ctx;
    protected:
      std::vector<std::pair<Operation*,GenerationID> > operations; 
      // We also need a data structure to record when there are
      // aliased but non-interfering region requirements. This should
      // be pretty sparse so we'll make it a map
      std::map<unsigned,LegionVector<AliasChildren>::aligned> aliased_children;
      TracingState state;
      // Pointer to a physical trace
      PhysicalTrace *physical_trace;
      unsigned last_memoized;
      std::set<std::pair<Operation*,GenerationID> > frontiers;
#ifdef LEGION_SPY
    protected:
      std::map<std::pair<Operation*,GenerationID>,UniqueID> current_uids;
      std::map<std::pair<Operation*,GenerationID>,unsigned> num_regions;
#endif
    };

    /**
     * \class StaticTrace
     * A static trace is a trace object that is used for 
     * handling cases where the application knows the dependneces
     * for a trace of operations
     */
    class StaticTrace : public LegionTrace,
                        public LegionHeapify<StaticTrace> {
    public:
      static const AllocationType alloc_type = STATIC_TRACE_ALLOC;
    public:
      StaticTrace(TaskContext *ctx, const std::set<RegionTreeID> *trees);
      StaticTrace(const StaticTrace &rhs);
      virtual ~StaticTrace(void);
    public:
      StaticTrace& operator=(const StaticTrace &rhs);
    public:
      virtual bool is_static_trace(void) const { return true; }
      virtual bool is_dynamic_trace(void) const { return false; }
      virtual StaticTrace* as_static_trace(void) { return this; }
      virtual DynamicTrace* as_dynamic_trace(void) { return NULL; }
      virtual TraceID get_trace_id(void) const { return 0; }
    public:
      virtual bool is_fixed(void) const;
      virtual bool handles_region_tree(RegionTreeID tid) const;
      virtual void record_static_dependences(Operation *op,
                              const std::vector<StaticDependence> *dependences);
      virtual void register_operation(Operation *op, GenerationID gen); 
      virtual void record_dependence(Operation *target,GenerationID target_gen,
                                     Operation *source,GenerationID source_gen);
      virtual void record_region_dependence(
                                    Operation *target, GenerationID target_gen,
                                    Operation *source, GenerationID source_gen,
                                    unsigned target_idx, unsigned source_idx,
                                    DependenceType dtype, bool validates,
                                    const FieldMask &dependent_mask);
      virtual void record_aliased_children(unsigned req_index, unsigned depth,
                                           const FieldMask &aliased_mask);
    protected:
      const LegionVector<DependenceRecord>::aligned&
                  translate_dependence_records(Operation *op, unsigned index);
    protected:
      std::deque<std::vector<StaticDependence> > static_dependences;
      std::deque<LegionVector<DependenceRecord>::aligned> translated_deps;
      std::set<RegionTreeID> application_trees;
    };

    /**
     * \class DynamicTrace
     * This class is used for memoizing the dynamic
     * dependence analysis for series of operations
     * in a given task's context.
     */
    class DynamicTrace : public LegionTrace,
                         public LegionHeapify<DynamicTrace> {
    public:
      static const AllocationType alloc_type = DYNAMIC_TRACE_ALLOC;
    public:
      struct OperationInfo {
      public:
        OperationInfo(Operation *op)
          : kind(op->get_operation_kind()), count(op->get_region_count()) { }
      public:
        Operation::OpKind kind;
        unsigned count;
      }; 
    public:
      DynamicTrace(TraceID tid, TaskContext *ctx, bool logical_only);
      DynamicTrace(const DynamicTrace &rhs);
      virtual ~DynamicTrace(void);
    public:
      DynamicTrace& operator=(const DynamicTrace &rhs);
    public:
      virtual bool is_static_trace(void) const { return false; }
      virtual bool is_dynamic_trace(void) const { return true; }
      virtual StaticTrace* as_static_trace(void) { return NULL; }
      virtual DynamicTrace* as_dynamic_trace(void) { return this; }
      virtual TraceID get_trace_id(void) const { return tid; }
    public:
      // Called by task execution thread
      virtual bool is_fixed(void) const { return fixed; }
      void fix_trace(void);
    public:
      // Called by analysis thread
      void end_trace_capture(void);
    public:
      virtual void record_static_dependences(Operation *op,
                          const std::vector<StaticDependence> *dependences);
      virtual bool handles_region_tree(RegionTreeID tid) const;
      // Called by analysis thread
      virtual void register_operation(Operation *op, GenerationID gen);
      virtual void record_dependence(Operation *target,GenerationID target_gen,
                                     Operation *source,GenerationID source_gen);
      virtual void record_region_dependence(
                                    Operation *target, GenerationID target_gen,
                                    Operation *source, GenerationID source_gen,
                                    unsigned target_idx, unsigned source_idx,
                                    DependenceType dtype, bool validates,
                                    const FieldMask &dependent_mask);
      virtual void record_aliased_children(unsigned req_index, unsigned depth,
                                           const FieldMask &aliased_mask);
    protected:
      // Insert a normal dependence for the current operation
      void insert_dependence(const DependenceRecord &record);
      // Insert an internal dependence for given key
      void insert_dependence(const std::pair<InternalOp*,GenerationID> &key,
                             const DependenceRecord &record);
    protected:
      // Only need this backwards lookup for recording dependences
      std::map<std::pair<Operation*,GenerationID>,unsigned> op_map;
      // Internal operations have a nasty interaction with traces because
      // we can generate different sets of internal operations each time we
      // run the trace depending on the state of the logical region tree.
      // Therefore, we keep track of internal ops done when capturing the trace
      // record transitive dependences on the other operations in the
      // trace that we would have interfered with had the internal operations
      // not been necessary.
      std::map<std::pair<InternalOp*,GenerationID>,
               LegionVector<DependenceRecord>::aligned> internal_dependences;
    protected: 
      // This is the generalized form of the dependences
      // For each operation, we remember a list of operations that
      // it dependens on and whether it is a validates the region
      std::deque<LegionVector<DependenceRecord>::aligned> dependences;
      // Metadata for checking the validity of a trace when it is replayed
      std::vector<OperationInfo> op_info;
    protected:
      const TraceID tid;
      bool fixed;
      bool tracing;
    };

    class TraceOp : public FenceOp {
    public:
      TraceOp(Runtime *rt);
      TraceOp(const TraceOp &rhs);
      virtual ~TraceOp(void);
    public:
      TraceOp& operator=(const TraceOp &rhs);
    public:
      virtual void execute_dependence_analysis(void);
    protected:
      LegionTrace *local_trace;
    };

    /**
     * \class TraceCaptureOp
     * This class represents trace operations which we inject
     * into the operation stream to mark when a trace capture
     * is finished so the DynamicTrace object can compute the
     * dependences data structure.
     */
    class TraceCaptureOp : public TraceOp {
    public:
      static const AllocationType alloc_type = TRACE_CAPTURE_OP_ALLOC;
    public:
      TraceCaptureOp(Runtime *rt);
      TraceCaptureOp(const TraceCaptureOp &rhs);
      virtual ~TraceCaptureOp(void);
    public:
      TraceCaptureOp& operator=(const TraceCaptureOp &rhs);
    public:
      void initialize_capture(TaskContext *ctx);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
    protected:
      DynamicTrace *dynamic_trace;
      PhysicalTemplate *current_template;
    };

    /**
     * \class TraceCompleteOp
     * This class represents trace operations which we inject
     * into the operation stream to mark when the execution
     * of a trace has been completed.  This fence operation
     * then registers dependences on all operations in the trace
     * and becomes the new current fence.
     */
    class TraceCompleteOp : public TraceOp {
    public:
      static const AllocationType alloc_type = TRACE_COMPLETE_OP_ALLOC;
    public:
      TraceCompleteOp(Runtime *rt);
      TraceCompleteOp(const TraceCompleteOp &rhs);
      virtual ~TraceCompleteOp(void);
    public:
      TraceCompleteOp& operator=(const TraceCompleteOp &rhs);
    public:
      void initialize_complete(TaskContext *ctx);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
    protected:
      PhysicalTemplate *current_template;
      ApEvent template_completion;
      bool replayed;
    };

    /**
     * \class TraceReplayOp
     * This class represents trace operations which we inject
     * into the operation stream to replay a physical trace
     * if there is one that satisfies its preconditions.
     */
    class TraceReplayOp : public TraceOp {
    public:
      static const AllocationType alloc_type = TRACE_REPLAY_OP_ALLOC;
    public:
      TraceReplayOp(Runtime *rt);
      TraceReplayOp(const TraceReplayOp &rhs);
      virtual ~TraceReplayOp(void);
    public:
      TraceReplayOp& operator=(const TraceReplayOp &rhs);
    public:
      void initialize_replay(TaskContext *ctx, LegionTrace *trace);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
    };

    /**
     * \class TraceBeginOp
     * This class represents mapping fences which we inject
     * into the operation stream to begin a trace.  This fence
     * is by a TraceReplayOp if the trace allows physical tracing.
     */
    class TraceBeginOp : public TraceOp {
    public:
      static const AllocationType alloc_type = TRACE_BEGIN_OP_ALLOC;
    public:
      TraceBeginOp(Runtime *rt);
      TraceBeginOp(const TraceBeginOp &rhs);
      virtual ~TraceBeginOp(void);
    public:
      TraceBeginOp& operator=(const TraceBeginOp &rhs);
    public:
      void initialize_begin(TaskContext *ctx, LegionTrace *trace);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    };

    struct CachedMapping
    {
      VariantID               chosen_variant;
      TaskPriority            task_priority;
      bool                    postmap_task;
      std::vector<Processor>  target_procs;
      std::deque<InstanceSet> physical_instances;
    };

    typedef
      LegionMap<std::pair<unsigned, DomainPoint>, CachedMapping>::aligned
      CachedMappings;

    /**
     * \class PhysicalTrace
     * This class is used for memoizing the dynamic physical dependence
     * analysis for series of operations in a given task's context.
     */
    class PhysicalTrace {
    public:
      PhysicalTrace(Runtime *runtime, LegionTrace *logical_trace);
      PhysicalTrace(const PhysicalTrace &rhs);
      ~PhysicalTrace(void);
    public:
      PhysicalTrace& operator=(const PhysicalTrace &rhs);
    public:
      void clear_cached_template(void) { current_template = NULL; }
      void invalidate_current_template(void);
      void check_template_preconditions(void);
    public:
      PhysicalTemplate* get_current_template(void) { return current_template; }
      bool has_any_templates(void) const { return templates.size() > 0; }
    public:
      PhysicalTemplate* start_new_template(ApEvent fence_event);
      void fix_trace(PhysicalTemplate *tpl);
    public:
      void initialize_template(ApEvent fence_completion, bool recurrent);
    public:
      const Runtime *runtime;
      const LegionTrace *logical_trace;
    private:
      mutable LocalLock trace_lock;
      PhysicalTemplate* current_template;
      std::vector<PhysicalTemplate*> templates;
      unsigned nonreplayable_count;
    };

    typedef Memoizable::TraceLocalID TraceLocalID;

    /**
     * \class PhysicalTemplate
     * This class represents a recipe to reconstruct a physical task graph.
     * A template consists of a sequence of instructions, each of which is
     * interpreted by the template engine. The template also maintains
     * the interpreter state (operations and events). These are initialized
     * before the template gets executed.
     */
    struct PhysicalTemplate {
    public:
      PhysicalTemplate(ApEvent fence_event);
      PhysicalTemplate(const PhysicalTemplate &rhs);
    private:
      friend class PhysicalTrace;
      ~PhysicalTemplate(void);
    public:
      void initialize(ApEvent fence_completion, bool recurrent);
      ApEvent get_completion(void) const;
    private:
      static bool check_logical_open(RegionTreeNode *node, ContextID ctx,
                                     FieldMask fields);
      static bool check_logical_open(RegionTreeNode *node, ContextID ctx,
                          LegionMap<IndexSpaceNode*, FieldMask>::aligned projs);
    public:
      bool check_preconditions(void);
      bool check_replayable(void);
      void register_operation(Operation *op);
      void execute_all(void);
      void finalize(void);
      void optimize(void);
      void dump_template(void);
    public:
      inline bool is_recording(void) const { return recording; }
      inline bool is_replaying(void) const { return !recording; }
      inline bool is_replayable(void) const { return replayable; }
    protected:
      static std::string view_to_string(const InstanceView *view);
      static std::string view_to_string(const FillView *view);
      void sanity_check(void);
    public:
      void record_mapper_output(SingleTask *task,
                                const Mapper::MapTaskOutput &output,
                             const std::deque<InstanceSet> &physical_instances);
      void get_mapper_output(SingleTask *task,
                             VariantID &chosen_variant,
                             TaskPriority &task_priority,
                             bool &postmap_task,
                             std::vector<Processor> &target_proc,
                             std::deque<InstanceSet> &physical_instances) const;
    public:
      void record_get_term_event(ApEvent lhs, SingleTask* task);
      void record_create_ap_user_event(ApUserEvent lhs);
      void record_trigger_event(ApUserEvent lhs, ApEvent rhs);
      void record_merge_events(ApEvent &lhs, ApEvent rhs);
      void record_merge_events(ApEvent &lhs, ApEvent e1, ApEvent e2);
      void record_merge_events(ApEvent &lhs, ApEvent e1, ApEvent e2,
                               ApEvent e3);
      void record_merge_events(ApEvent &lhs, const std::set<ApEvent>& rhs);
      void record_copy_views(InstanceView *src,
                             const FieldMask &src_mask,
                             ContextID src_logical_ctx,
                             ContextID src_physucal_ctx,
                             InstanceView *dst,
                             const FieldMask &dst_mask,
                             ContextID dst_logical_ctx,
                             ContextID dst_physical_ctx);
      void record_issue_copy(Operation* op, ApEvent &lhs,
                             RegionNode *node,
                             const std::vector<CopySrcDstField>& src_fields,
                             const std::vector<CopySrcDstField>& dst_fields,
                             ApEvent precondition,
                             PredEvent predicate_guard,
                             RegionTreeNode *intersect,
                             ReductionOpID redop,
                             bool reduction_fold);
      void record_empty_copy(CompositeView *src,
                             const FieldMask &src_mask,
                             MaterializedView *dst,
                             const FieldMask &dst_mask,
                             ContextID logical_ctx);
      void record_set_ready_event(Operation *op,
                                  unsigned region_idx,
                                  unsigned inst_idx,
                                  ApEvent &ready_event,
                                  const RegionRequirement &req,
                                  InstanceView *view,
                                  const FieldMask &fields,
                                  ContextID logical_ctx,
                                  ContextID physical_ctx);
      void record_get_op_term_event(ApEvent lhs, Operation *op);
      void record_set_op_sync_event(ApEvent &lhs, Operation *op);
      void record_complete_replay(Operation *op, ApEvent rhs);
      void record_issue_fill(Operation *op, ApEvent &lhs,
                             RegionNode *node,
                             const std::vector<CopySrcDstField> &fields,
                             const void *fill_buffer, size_t fill_size,
                             ApEvent precondition,
                             PredEvent predicate_guard,
#ifdef LEGION_SPY
                             UniqueID fill_uid,
#endif
                             RegionTreeNode *intersect);

      void record_fill_view(FillView *fill_view, const FieldMask &fill_mask);
      void record_deferred_copy_from_fill_view(FillView *fill_view,
                                               InstanceView *dst_view,
                                               const FieldMask &copy_mask,
                                               ContextID logical_ctx,
                                               ContextID physical_ctx);
      void record_empty_copy_from_fill_view(InstanceView *dst_view,
                                            const FieldMask &copy_mask,
                                            ContextID logical_ctx,
                                            ContextID physical_ctx);
      void record_blocking_call(void);
      void record_outstanding_gc_event(InstanceView *view, ApEvent term_event);
    private:
      void record_ready_view(const RegionRequirement &req,
                             InstanceView *view,
                             const FieldMask &fields,
                             ContextID logical_ctx,
                             ContextID physical_ctx);
      void record_last_user(const PhysicalInstance &inst, RegionNode *node,
                            unsigned field, unsigned user, bool read);
      void find_last_users(const PhysicalInstance &inst, unsigned field,
                           std::set<unsigned> &users);
    private:
      bool recording;
      bool replayable;
      bool has_block;
      mutable LocalLock template_lock;
      unsigned fence_completion_id;
    private:
      std::map<ApEvent, unsigned> event_map;
      std::vector<Instruction*> instructions;
      std::map<TraceLocalID, unsigned> task_entries;
      typedef std::pair<PhysicalInstance, unsigned> InstanceAccess;
      struct UserInfo {
        std::set<unsigned> users;
        std::set<RegionNode*> nodes;
        bool read;
      };
      std::map<InstanceAccess, UserInfo> last_users;
      std::map<unsigned, unsigned> frontiers;
    public:
      ApEvent fence_completion;
      std::map<TraceLocalID, Operation*> operations;
      std::vector<ApEvent> events;
      std::vector<ApUserEvent> user_events;
      CachedMappings                                  cached_mappings;
      LegionMap<InstanceView*, FieldMask>::aligned    previous_valid_views;
      LegionMap<std::pair<RegionTreeNode*, ContextID>,
                FieldMask>::aligned                   previous_open_nodes;
      std::map<std::pair<RegionTreeNode*, ContextID>,
               LegionMap<IndexSpaceNode*, FieldMask>::aligned>
                                                      previous_projections;
      LegionMap<InstanceView*, FieldMask>::aligned    valid_views;
      LegionMap<InstanceView*, FieldMask>::aligned    reduction_views;
      LegionMap<FillView*,     FieldMask>::aligned    fill_views;
      LegionMap<FillView*,     FieldMask>::aligned    untracked_fill_views;
      LegionMap<InstanceView*, bool>::aligned         initialized;
      LegionMap<InstanceView*, ContextID>::aligned    logical_contexts;
      LegionMap<InstanceView*, ContextID>::aligned    physical_contexts;
      std::map<InstanceView*, std::set<ApEvent> >     outstanding_gc_events;
    };

    enum InstructionKind
    {
      GET_TERM_EVENT = 0,
      GET_OP_TERM_EVENT,
      CREATE_AP_USER_EVENT,
      TRIGGER_EVENT,
      MERGE_EVENT,
      ISSUE_COPY,
      ISSUE_FILL,
      SET_OP_SYNC_EVENT,
      ASSIGN_FENCE_COMPLETION,
      COMPLETE_REPLAY,
    };

    /**
     * \class Instruction
     * This class is an abstract parent class for all template instructions.
     */
    struct Instruction {
      Instruction(PhysicalTemplate& tpl);
      virtual ~Instruction(void) {};
      virtual void execute(void) = 0;
      virtual std::string to_string(void) = 0;

      virtual InstructionKind get_kind(void) = 0;
      virtual GetTermEvent* as_get_term_event(void) = 0;
      virtual CreateApUserEvent* as_create_ap_user_event(void) = 0;
      virtual TriggerEvent* as_trigger_event(void) = 0;
      virtual MergeEvent* as_merge_event(void) = 0;
      virtual AssignFenceCompletion* as_assignment_fence_completion(void) = 0;
      virtual IssueCopy* as_issue_copy(void) = 0;
      virtual IssueFill* as_issue_fill(void) = 0;
      virtual GetOpTermEvent* as_get_op_term_event(void) = 0;
      virtual SetOpSyncEvent* as_set_op_sync_event(void) = 0;
      virtual CompleteReplay* as_complete_replay(void) = 0;

      virtual Instruction* clone(PhysicalTemplate& tpl,
                               const std::map<unsigned, unsigned> &rewrite) = 0;

      virtual TraceLocalID get_owner(const TraceLocalID &key) = 0;

    protected:
      std::map<TraceLocalID, Operation*> &operations;
      std::vector<ApEvent> &events;
      std::vector<ApUserEvent> &user_events;
    };

    /**
     * \class GetTermEvent
     * This instruction has the following semantics:
     *   events[lhs] = operations[rhs].get_task_completion()
     */
    struct GetTermEvent : public Instruction {
      GetTermEvent(PhysicalTemplate& tpl, unsigned lhs,
                   const TraceLocalID& rhs);
      virtual void execute(void);
      virtual std::string to_string(void);

      virtual InstructionKind get_kind(void)
        { return GET_TERM_EVENT; }
      virtual GetTermEvent* as_get_term_event(void)
        { return this; }
      virtual CreateApUserEvent* as_create_ap_user_event(void)
        { return NULL; }
      virtual TriggerEvent* as_trigger_event(void)
        { return NULL; }
      virtual MergeEvent* as_merge_event(void)
        { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion(void)
        { return NULL; }
      virtual IssueCopy* as_issue_copy(void)
        { return NULL; }
      virtual IssueFill* as_issue_fill(void)
        { return NULL; }
      virtual GetOpTermEvent* as_get_op_term_event(void)
        { return NULL; }
      virtual SetOpSyncEvent* as_set_op_sync_event(void)
        { return NULL; }
      virtual CompleteReplay* as_complete_replay(void)
        { return NULL; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalID get_owner(const TraceLocalID &key) { return rhs; }

    private:
      friend struct PhysicalTemplate;
      unsigned lhs;
      TraceLocalID rhs;
    };

    /**
     * \class CreateApUserEvent
     * This instruction has the following semantics:
     *   events[lhs] = Runtime::create_ap_user_event()
     */
    struct CreateApUserEvent : public Instruction {
      CreateApUserEvent(PhysicalTemplate& tpl, unsigned lhs);
      virtual void execute(void);
      virtual std::string to_string(void);

      virtual InstructionKind get_kind(void)
        { return CREATE_AP_USER_EVENT; }
      virtual GetTermEvent* as_get_term_event(void)
        { return NULL; }
      virtual CreateApUserEvent* as_create_ap_user_event(void)
        { return this; }
      virtual TriggerEvent* as_trigger_event(void)
        { return NULL; }
      virtual MergeEvent* as_merge_event(void)
        { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion(void)
        { return NULL; }
      virtual IssueCopy* as_issue_copy(void)
        { return NULL; }
      virtual IssueFill* as_issue_fill(void)
        { return NULL; }
      virtual GetOpTermEvent* as_get_op_term_event(void)
        { return NULL; }
      virtual SetOpSyncEvent* as_set_op_sync_event(void)
        { return NULL; }
      virtual CompleteReplay* as_complete_replay(void)
        { return NULL; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalID get_owner(const TraceLocalID &key) { return key; }

    private:
      friend struct PhysicalTemplate;
      unsigned lhs;
    };

    /**
     * \class TriggerEvent
     * This instruction has the following semantics:
     *   Runtime::trigger_event(events[lhs], events[rhs])
     */
    struct TriggerEvent : public Instruction {
      TriggerEvent(PhysicalTemplate& tpl, unsigned lhs, unsigned rhs);
      virtual void execute(void);
      virtual std::string to_string(void);

      virtual InstructionKind get_kind(void)
        { return TRIGGER_EVENT; }
      virtual GetTermEvent* as_get_term_event(void)
        { return NULL; }
      virtual CreateApUserEvent* as_create_ap_user_event(void)
        { return NULL; }
      virtual TriggerEvent* as_trigger_event(void)
        { return this; }
      virtual MergeEvent* as_merge_event(void)
        { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion(void)
        { return NULL; }
      virtual IssueCopy* as_issue_copy(void)
        { return NULL; }
      virtual IssueFill* as_issue_fill(void)
        { return NULL; }
      virtual GetOpTermEvent* as_get_op_term_event(void)
        { return NULL; }
      virtual SetOpSyncEvent* as_set_op_sync_event(void)
        { return NULL; }
      virtual CompleteReplay* as_complete_replay(void)
        { return NULL; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalID get_owner(const TraceLocalID &key) { return key; }

    private:
      friend struct PhysicalTemplate;
      unsigned lhs;
      unsigned rhs;
    };

    /**
     * \class MergeEvent
     * This instruction has the following semantics:
     *   events[lhs] = Runtime::merge_events(events[rhs])
     */
    struct MergeEvent : public Instruction {
      MergeEvent(PhysicalTemplate& tpl, unsigned lhs,
                 const std::set<unsigned>& rhs);
      virtual void execute(void);
      virtual std::string to_string(void);

      virtual InstructionKind get_kind(void)
        { return MERGE_EVENT; }
      virtual GetTermEvent* as_get_term_event(void)
        { return NULL; }
      virtual CreateApUserEvent* as_create_ap_user_event(void)
        { return NULL; }
      virtual TriggerEvent* as_trigger_event(void)
        { return NULL; }
      virtual MergeEvent* as_merge_event(void)
        { return this; }
      virtual AssignFenceCompletion* as_assignment_fence_completion(void)
        { return NULL; }
      virtual IssueCopy* as_issue_copy(void)
        { return NULL; }
      virtual IssueFill* as_issue_fill(void)
        { return NULL; }
      virtual GetOpTermEvent* as_get_op_term_event(void)
        { return NULL; }
      virtual SetOpSyncEvent* as_set_op_sync_event(void)
        { return NULL; }
      virtual CompleteReplay* as_complete_replay(void)
        { return NULL; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalID get_owner(const TraceLocalID &key) { return key; }

    private:
      friend struct PhysicalTemplate;
      unsigned lhs;
      std::set<unsigned> rhs;
    };

    /**
     * \class AssignFenceCompletion
     * This instruction has the following semantics:
     *   events[lhs] = fence_completion
     */
    struct AssignFenceCompletion : public Instruction {
      AssignFenceCompletion(PhysicalTemplate& tpl, unsigned lhs);
      virtual void execute(void);
      virtual std::string to_string(void);

      virtual InstructionKind get_kind(void)
        { return ASSIGN_FENCE_COMPLETION; }
      virtual GetTermEvent* as_get_term_event(void)
        { return NULL; }
      virtual CreateApUserEvent* as_create_ap_user_event(void)
        { return NULL; }
      virtual TriggerEvent* as_trigger_event(void)
        { return NULL; }
      virtual MergeEvent* as_merge_event(void)
        { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion(void)
        { return this; }
      virtual IssueCopy* as_issue_copy(void)
        { return NULL; }
      virtual IssueFill* as_issue_fill(void)
        { return NULL; }
      virtual GetOpTermEvent* as_get_op_term_event(void)
        { return NULL; }
      virtual SetOpSyncEvent* as_set_op_sync_event(void)
        { return NULL; }
      virtual CompleteReplay* as_complete_replay(void)
        { return NULL; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalID get_owner(const TraceLocalID &key) { return key; }

    private:
      friend struct PhysicalTemplate;
      ApEvent &fence_completion;
      unsigned lhs;
    };

    /**
     * \class IssueFill
     * This instruction has the following semantics:
     *
     *   events[lhs] = domain.fill(fields, requests,
     *                             fill_buffer, fill_size,
     *                             events[precondition_idx]);
     */
    struct IssueFill : public Instruction {
      IssueFill(PhysicalTemplate& tpl,
                unsigned lhs, RegionNode *node,
                const TraceLocalID &op_key,
                const std::vector<CopySrcDstField> &fields,
                const void *fill_buffer, size_t fill_size,
                unsigned precondition_idx,
                PredEvent predicate_guard,
#ifdef LEGION_SPY
                UniqueID fill_uid,
#endif
                RegionTreeNode *intersect);
      virtual ~IssueFill(void);
      virtual void execute(void);
      virtual std::string to_string(void);

      virtual InstructionKind get_kind(void)
        { return ISSUE_FILL; }
      virtual GetTermEvent* as_get_term_event(void)
        { return NULL; }
      virtual CreateApUserEvent* as_create_ap_user_event(void)
        { return NULL; }
      virtual TriggerEvent* as_trigger_event(void)
        { return NULL; }
      virtual MergeEvent* as_merge_event(void)
        { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion(void)
        { return NULL; }
      virtual IssueCopy* as_issue_copy(void)
        { return NULL; }
      virtual IssueFill* as_issue_fill(void)
        { return this; }
      virtual GetOpTermEvent* as_get_op_term_event(void)
        { return NULL; }
      virtual SetOpSyncEvent* as_set_op_sync_event(void)
        { return NULL; }
      virtual CompleteReplay* as_complete_replay(void)
        { return NULL; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalID get_owner(const TraceLocalID &key) { return op_key; }

    private:
      friend struct PhysicalTemplate;
      unsigned lhs;
      RegionNode *node;
      TraceLocalID op_key;
      std::vector<CopySrcDstField> fields;
      void *fill_buffer;
      size_t fill_size;
      unsigned precondition_idx;
      PredEvent predicate_guard;
#ifdef LEGION_SPY
      UniqueID fill_uid;
#endif
      RegionTreeNode *intersect;
    };

    /**
     * \class IssueCopy
     * This instruction has the following semantics:
     *   events[lhs] = node->issue_copy(operations[op_key],
     *                                  src_fields, dst_fields,
     *                                  events[precondition_idx],
     *                                  predicate_guard, intersect,
     *                                  redop, reduction_fold);
     */
    struct IssueCopy : public Instruction {
      IssueCopy(PhysicalTemplate &tpl,
                unsigned lhs, RegionNode *node,
                const TraceLocalID &op_key,
                const std::vector<CopySrcDstField>& src_fields,
                const std::vector<CopySrcDstField>& dst_fields,
                unsigned precondition_idx, PredEvent predicate_guard,
                RegionTreeNode *intersect,
                ReductionOpID redop, bool reduction_fold);
      virtual void execute(void);
      virtual std::string to_string(void);

      virtual InstructionKind get_kind(void)
        { return ISSUE_COPY; }
      virtual GetTermEvent* as_get_term_event(void)
        { return NULL; }
      virtual CreateApUserEvent* as_create_ap_user_event(void)
        { return NULL; }
      virtual TriggerEvent* as_trigger_event(void)
        { return NULL; }
      virtual MergeEvent* as_merge_event(void)
        { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion(void)
        { return NULL; }
      virtual IssueCopy* as_issue_copy(void)
        { return this; }
      virtual IssueFill* as_issue_fill(void)
        { return NULL; }
      virtual GetOpTermEvent* as_get_op_term_event(void)
        { return NULL; }
      virtual SetOpSyncEvent* as_set_op_sync_event(void)
        { return NULL; }
      virtual CompleteReplay* as_complete_replay(void)
        { return NULL; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalID get_owner(const TraceLocalID &key) { return op_key; }

    private:
      friend struct PhysicalTemplate;
      unsigned lhs;
      RegionNode *node;
      TraceLocalID op_key;
      std::vector<CopySrcDstField> src_fields;
      std::vector<CopySrcDstField> dst_fields;
      unsigned precondition_idx;
      PredEvent predicate_guard;
      RegionTreeNode *intersect;
      ReductionOpID redop;
      bool reduction_fold;
    };

    /**
     * \class GetOpTermEvent
     * This instruction has the following semantics:
     *   events[lhs] = operations[rhs].get_completion_event()
     */
    struct GetOpTermEvent : public Instruction {
      GetOpTermEvent(PhysicalTemplate& tpl, unsigned lhs,
                       const TraceLocalID& rhs);
      virtual void execute(void);
      virtual std::string to_string(void);

      virtual InstructionKind get_kind(void)
        { return GET_OP_TERM_EVENT; }
      virtual GetTermEvent* as_get_term_event(void)
        { return NULL; }
      virtual CreateApUserEvent* as_create_ap_user_event(void)
        { return NULL; }
      virtual TriggerEvent* as_trigger_event(void)
        { return NULL; }
      virtual MergeEvent* as_merge_event(void)
        { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion(void)
        { return NULL; }
      virtual IssueCopy* as_issue_copy(void)
        { return NULL; }
      virtual IssueFill* as_issue_fill(void)
        { return NULL; }
      virtual GetOpTermEvent* as_get_op_term_event(void)
        { return this; }
      virtual SetOpSyncEvent* as_set_op_sync_event(void)
        { return NULL; }
      virtual CompleteReplay* as_complete_replay(void)
        { return NULL; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalID get_owner(const TraceLocalID &key) { return rhs; }

    private:
      friend struct PhysicalTemplate;
      unsigned lhs;
      TraceLocalID rhs;
    };

    /**
     * \class SetOpSyncEvent
     * This instruction has the following semantics:
     *   events[lhs] = operations[rhs].compute_sync_precondition()
     */
    struct SetOpSyncEvent : public Instruction {
      SetOpSyncEvent(PhysicalTemplate& tpl, unsigned lhs,
                       const TraceLocalID& rhs);
      virtual void execute(void);
      virtual std::string to_string(void);

      virtual InstructionKind get_kind(void)
        { return SET_OP_SYNC_EVENT; }
      virtual GetTermEvent* as_get_term_event(void)
        { return NULL; }
      virtual CreateApUserEvent* as_create_ap_user_event(void)
        { return NULL; }
      virtual TriggerEvent* as_trigger_event(void)
        { return NULL; }
      virtual MergeEvent* as_merge_event(void)
        { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion(void)
        { return NULL; }
      virtual IssueCopy* as_issue_copy(void)
        { return NULL; }
      virtual IssueFill* as_issue_fill(void)
        { return NULL; }
      virtual GetOpTermEvent* as_get_op_term_event(void)
        { return NULL; }
      virtual SetOpSyncEvent* as_set_op_sync_event(void)
        { return this; }
      virtual CompleteReplay* as_complete_replay(void)
        { return NULL; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalID get_owner(const TraceLocalID &key) { return rhs; }

    private:
      friend struct PhysicalTemplate;
      unsigned lhs;
      TraceLocalID rhs;
    };

    /**
     * \class CompleteReplay
     * This instruction has the following semantics:
     *   operations[lhs]->complete_replay(events[rhs])
     */
    struct CompleteReplay : public Instruction {
      CompleteReplay(PhysicalTemplate& tpl, const TraceLocalID& lhs,
                            unsigned rhs);
      virtual void execute(void);
      virtual std::string to_string(void);

      virtual InstructionKind get_kind(void)
        { return COMPLETE_REPLAY; }
      virtual GetTermEvent* as_get_term_event(void)
        { return NULL; }
      virtual CreateApUserEvent* as_create_ap_user_event(void)
        { return NULL; }
      virtual TriggerEvent* as_trigger_event(void)
        { return NULL; }
      virtual MergeEvent* as_merge_event(void)
        { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion(void)
        { return NULL; }
      virtual IssueCopy* as_issue_copy(void)
        { return NULL; }
      virtual IssueFill* as_issue_fill(void)
        { return NULL; }
      virtual GetOpTermEvent* as_get_op_term_event(void)
        { return NULL; }
      virtual SetOpSyncEvent* as_set_op_sync_event(void)
        { return NULL; }
      virtual CompleteReplay* as_complete_replay(void)
        { return this; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalID get_owner(const TraceLocalID &key) { return lhs; }

    private:
      friend struct PhysicalTemplate;
      TraceLocalID lhs;
      unsigned rhs;
    };

  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_TRACE__

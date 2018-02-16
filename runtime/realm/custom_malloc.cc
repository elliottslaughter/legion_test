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

// custom memory allocation in Realm

#include "realm/custom_malloc.h"
#include "realm/timers.h"
#include "realm/numa/numasysif.h"

#include "realm/faults.h"
#include <set>

#include <sys/types.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <stdio.h>
#include <math.h>

#define SANITY_CHECK_ALLOCATIONS_IN_RANGE

extern "C" {
  void *__libc_malloc(size_t size);
  void *__libc_memalign(size_t alignment, size_t size);
  void *__libc_realloc(void *ptr, size_t size);
  void __libc_free(void *ptr);
};

// we can't use assert() in this code because the failure path calls malloc
//  (at least with glibc), and infinite loops are not your friend
#define NONMALLOC_ASSERT(cond) \
  do {									\
    if(__builtin_expect((cond) == 0, 0))				\
      nonmalloc_assert_fail(__FILE__, __LINE__, #cond);			\
  } while(0)

__attribute__((noreturn))
static void nonmalloc_assert_fail(const char *file, unsigned line, const char *cond)
{
  char msg[512];
  int len = snprintf(msg, 512, "%s:%u: Assertion `%s' failed.\n", file, line, cond);
  fwrite(msg, 1, len, stderr);
  fflush(stderr);
  abort();
}

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // struct MemoryRanges

  struct MemoryRanges {
    MemoryRanges(void);

    static const size_t MAX_MEMORY_RANGES = 64;

    uintptr_t starts[MAX_MEMORY_RANGES];
    uintptr_t ends[MAX_MEMORY_RANGES];
    Allocator *allocators[MAX_MEMORY_RANGES];
    size_t num_ranges;
    Allocator *default_allocator;
  };

  static MemoryRanges static_ranges;
  static MemoryRanges *ranges = 0;

  MemoryRanges::MemoryRanges(void)
    : num_ranges(0)
    , default_allocator(0)
  {
    memset(starts, 0, sizeof(uintptr_t) * MAX_MEMORY_RANGES);
    memset(ends, 0, sizeof(uintptr_t) * MAX_MEMORY_RANGES);
    memset(allocators, 0, sizeof(Allocator *) * MAX_MEMORY_RANGES);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Allocator

  Allocator::~Allocator(void)
  {}

  /*static*/ Allocator *Allocator::find_allocator(void *ptr)
  {
    // if we don't have a range list yet, nothing to find
    if(!ranges) return 0;

    // linear search should be faster for small numbers of ranges - maybe
    //  vectorizable
    for(size_t i = 0; i < ranges->num_ranges; i++) {
      if(reinterpret_cast<uintptr_t>(ptr) < ranges->starts[i]) continue;
      if(reinterpret_cast<uintptr_t>(ptr) >= ranges->ends[i]) continue;
      return ranges->allocators[i];
    }
    return ranges->default_allocator;
  }

  /*static*/ void Allocator::register_memory_range(Allocator *alloc, void *base, size_t bytes)
  {
    NONMALLOC_ASSERT(ranges != 0);

    size_t idx = __sync_fetch_and_add(&ranges->num_ranges, 1);
    NONMALLOC_ASSERT(idx < MemoryRanges::MAX_MEMORY_RANGES);
    
    // a range cannot be matched until the end value becomes nonzero, so
    //  write starts and allocators before writing ends
    ranges->starts[idx] = reinterpret_cast<uintptr_t>(base);
    ranges->allocators[idx] = alloc;
    ranges->ends[idx] = reinterpret_cast<uintptr_t>(base) + bytes;
  }

  /*static*/ void Allocator::unregister_memory_range(Allocator *alloc, void *base, size_t bytes)
  {
    NONMALLOC_ASSERT(ranges != 0);

    // find the range and zero it out - don't try to reuse it for now
    for(size_t i = 0; i < ranges->num_ranges; i++) {
      if(ranges->starts[i] != reinterpret_cast<uintptr_t>(base)) continue;
      // end and allocator pointer must match if start did
      NONMALLOC_ASSERT(ranges->ends[i] == uintptr_t(reinterpret_cast<uintptr_t>(base) + bytes));
      NONMALLOC_ASSERT(ranges->allocators[i] == alloc);
      // zero out ends first to avoid race conditions
      ranges->ends[i] = 0;
      ranges->starts[i] = 0;
      ranges->allocators[i] = 0;
      return;
    }
    NONMALLOC_ASSERT(0);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class LibcMalloc

  class LibcMalloc : public Allocator {
  public:
    virtual void *malloc(size_t size, size_t alignment);
    virtual void *realloc(void *oldptr, size_t new_size);
    virtual void free(void *ptr);
  };

  void *LibcMalloc::malloc(size_t size, size_t alignment)
  {
    if(size > 0) {
      if(alignment > 0)
	return __libc_memalign(alignment, size);
      else
	return __libc_malloc(size);
    } else
      return 0;
  }

  void *LibcMalloc::realloc(void *oldptr, size_t new_size)
  {
    if(oldptr != 0) {
      if(new_size > 0) {
	return __libc_realloc(oldptr, new_size);
      } else {
	__libc_free(oldptr);
	return 0;
      }
    } else {
      if(new_size > 0)
	return __libc_malloc(new_size);
      else
	return 0;
    }
  }

  void LibcMalloc::free(void *ptr)
  {
    if(ptr)
      __libc_free(ptr);
  }
  

  ////////////////////////////////////////////////////////////////////////
  //
  // class BumpAllocator

  class BumpAllocator : public Allocator {
  public:
    BumpAllocator(void);
    virtual ~BumpAllocator(void);

    void set_pool_location(void *base, size_t bytes);

    virtual void *malloc(size_t size, size_t alignment);
    virtual void *realloc(void *oldptr, size_t new_size);
    virtual void free(void *ptr);

    // this descriptor needs to be in shared memory too!
    struct PoolDescriptor {
      uintptr_t pool_start;
      uintptr_t pool_pos;
      uintptr_t pool_end;
    };
    PoolDescriptor pool;
    size_t pool_size;
  };

  BumpAllocator::BumpAllocator(void)
    : pool_size(0)
  {
#if 0
    void *mmap_base = mmap(0, pool_size,
			   PROT_READ | PROT_WRITE,
			   MAP_ANONYMOUS | MAP_SHARED | MAP_NORESERVE,
			   -1 /*fd*/, 0);
    NONMALLOC_ASSERT(mmap_base != reinterpret_cast<void *>(-1));
    pool = reinterpret_cast<PoolDescriptor *>(mmap_base);
    pool->pool_start = reinterpret_cast<uintptr_t>(mmap_base);
    pool->pool_pos = pool->pool_start + sizeof(PoolDescriptor);
    pool->pool_end = pool->pool_start + pool_size;
#endif
  }

  BumpAllocator::~BumpAllocator(void)
  {
#if 0
    int ret = munmap(pool, pool_size);
    NONMALLOC_ASSERT(ret == 0);
#endif
  }

  void BumpAllocator::set_pool_location(void *base, size_t bytes)
  {
    pool.pool_start = reinterpret_cast<uintptr_t>(base);
    pool.pool_pos = pool.pool_start;
    pool.pool_end = pool.pool_start + bytes;
  }

  void *BumpAllocator::malloc(size_t size, size_t alignment)
  {
    // if alignment is not specified, anything larger than 8B needs 16B
    //  alignment in 64-bit systems
    if((alignment == 0) && (size > 8)) alignment = 16;

    while(true) {
      uintptr_t cur_pos = __sync_fetch_and_add(&pool.pool_pos, 0);

      uintptr_t base = cur_pos + sizeof(size_t);
      if(alignment > 1) {
	uintptr_t leftover = base % alignment;
	if(leftover > 0) {
	  NONMALLOC_ASSERT((alignment % sizeof(size_t)) == 0);
	  base += alignment - leftover;
	}
      }
      (reinterpret_cast<size_t *>(base))[-1] = size;
      uintptr_t new_pos = base + size;
      // align to sizeof(size_t) if needed
      {
	uintptr_t leftover = size % sizeof(size_t);
	if(leftover > 0)
	  new_pos += sizeof(size_t) - leftover;
      }
      NONMALLOC_ASSERT(new_pos <= pool.pool_end);

      NONMALLOC_ASSERT((alignment <= 1) || ((base % alignment) == 0));

      // compare-and-swap to actually claim memory - loop around on failure
      if(__sync_bool_compare_and_swap(&pool.pool_pos, cur_pos, new_pos))
	return reinterpret_cast<void *>(base);
    }
  }

  void BumpAllocator::free(void *ptr)
  {
    // do nothing
  }

  void *BumpAllocator::realloc(void *oldptr, size_t new_size)
  {
    if(oldptr != 0) {
      // size of 0 is a free in disguise
      if(new_size == 0) {
	this->free(oldptr);
	return 0;
      }

      // get old size
      size_t old_size = (static_cast<size_t *>(oldptr))[-1];
      // shrinkage is easy
      if(new_size <= old_size) {
	(static_cast<size_t *>(oldptr))[-1] = new_size;
	return oldptr;
      }

      // otherwise malloc and copy
      void *newptr = this->malloc(new_size, 0);
      memcpy(newptr, oldptr, old_size);
      this->free(oldptr);
      return newptr;
    } else {
      if(new_size == 0)
	return 0;

      return this->malloc(new_size, 0);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class UsageTrackingAllocator

  template <typename T, size_t BUCKETS, size_t MINSIZE = 8, size_t SCALE=1024>
  class UsageTrackingAllocator : public T {
  public:
    UsageTrackingAllocator(void);

    virtual void *malloc(size_t size, size_t alignment);
    virtual void *realloc(void *oldptr, size_t new_size);
    virtual void free(void *ptr);

    void report(void);

  protected:
    static unsigned bucket_index(size_t size);

    struct AllocInfo {
      size_t size;
      unsigned short pad;
      unsigned short magic;
      static const unsigned short MAGIC_VALUE = 0x4921;
    };

    struct BucketInfo {
      BucketInfo(void);

      void record_alloc(size_t bytes, long long nanoseconds);
      void record_free(size_t bytes, long long nanoseconds);
      void report(void);

      size_t cur_allocs, max_allocs, total_allocs;
      size_t cur_bytes, max_bytes, total_bytes;
      long long sum_alloc_time, sum_alloc_time2, max_alloc_time;
      long long sum_free_time, sum_free_time2, max_free_time;
    };

    BucketInfo buckets[BUCKETS], global_stats;
    long long time_offset;
    GASNetHSL mutex;
    std::set<intptr_t> backtraces_seen;
  };

  template <typename T, size_t BUCKETS, size_t MINSIZE, size_t SCALE>
  UsageTrackingAllocator<T,BUCKETS,MINSIZE,SCALE>::UsageTrackingAllocator(void)
  {
    // take several measurements of the time offset between two clock
    //  measurement calls and take the minimum
    time_offset = 0;
    for(int i = 0; i < 10; i++) {
      long long t1 = Clock::current_time_in_nanoseconds();
      long long t2 = Clock::current_time_in_nanoseconds();
      long long elapsed = t2 - t1;
      if((i == 0) || (time_offset > elapsed))
	 time_offset = elapsed;
    }
    //printf("offset = %lld\n", time_offset);
  }

  template <typename T, size_t BUCKETS, size_t MINSIZE, size_t SCALE>
  void *UsageTrackingAllocator<T,BUCKETS,MINSIZE,SCALE>::malloc(size_t size, size_t alignment)
  {
    if(0&&(size <= 8 || size > 32768)) {
      AutoHSLLock al(mutex);
      Backtrace bt;
      bt.capture_backtrace();
      intptr_t h = bt.hash();
      if(backtraces_seen.count(h) == 0) {
	backtraces_seen.insert(h);
	bt.lookup_symbols();
	std::cout << "malloc of size=" << size << " from " << bt;
      }
    }
    // we need to squirrel away some space to keep the size info
    size_t pad = sizeof(AllocInfo);
    if((alignment > 0) && ((pad % alignment) != 0))
      pad = ((pad / alignment) + 1) * alignment;

    long long t1 = Clock::current_time_in_nanoseconds();
    void *ptr = T::malloc(size + pad, alignment);
    long long t2 = Clock::current_time_in_nanoseconds();
    if(!ptr) return 0;
    long long elapsed = t2 - t1 - time_offset;

    ptr = reinterpret_cast<char *>(ptr) + pad;
    AllocInfo *info = reinterpret_cast<AllocInfo *>(ptr) - 1;
    info->magic = AllocInfo::MAGIC_VALUE;
    info->size = size;
    info->pad = pad;
    //printf("alloc ptr=%p size=%zd pad=%zd\n", ptr, size, pad);

    unsigned idx = bucket_index(size);
    buckets[idx].record_alloc(size, elapsed);
    global_stats.record_alloc(size, elapsed);
    return ptr;
  }

  template <typename T, size_t BUCKETS, size_t MINSIZE, size_t SCALE>
  void *UsageTrackingAllocator<T,BUCKETS,MINSIZE,SCALE>::realloc(void *oldptr, size_t new_size)
  {
    if(oldptr != 0) {
      // get the old size and also check the magic value
      const AllocInfo *info = reinterpret_cast<AllocInfo *>(oldptr) - 1;
      if(info->magic != AllocInfo::MAGIC_VALUE) {
	//printf("%p is not our alloc?\n", oldptr);
	return T::realloc(oldptr, new_size);
      }

      if(new_size > 0) {
	void *newptr = this->malloc(new_size, 0);
	memcpy(newptr, oldptr, (info->size < new_size) ? info->size : new_size);
	this->free(oldptr);
	return newptr;
      } else {
	// just a free in disguise
	this->free(oldptr);
	return 0;
      }
    } else {
      if(new_size > 0) {
	// just a malloc
	return this->malloc(new_size, 0);
      } else {
	// nop
	return 0;
      }
    }
  }

  template <typename T, size_t BUCKETS, size_t MINSIZE, size_t SCALE>
  void UsageTrackingAllocator<T,BUCKETS,MINSIZE,SCALE>::free(void *ptr)
  {
    if(!ptr)
      return;

    // recover size and padding amounts - check magic value too
    const AllocInfo *info = reinterpret_cast<AllocInfo *>(ptr) - 1;
    if(info->magic != AllocInfo::MAGIC_VALUE) {
      //printf("%p is not our alloc?\n", ptr);
      T::free(ptr);
      return;
    }
    // save size since info is not accessible after free
    size_t size = info->size;

    //printf("free %p size=%zd pad=%d\n", ptr, info->size, info->pad);

    // move pointer back down to base of underlying alloc for free call
    ptr = reinterpret_cast<char *>(ptr) - info->pad;
    long long t1 = Clock::current_time_in_nanoseconds();
    T::free(ptr);
    long long t2 = Clock::current_time_in_nanoseconds();
    long long elapsed = t2 - t1 - time_offset;

    unsigned idx = bucket_index(size);
    buckets[idx].record_free(size, elapsed);
    global_stats.record_free(size, elapsed);

  }

  template <typename T, size_t BUCKETS, size_t MINSIZE, size_t SCALE>
  /*static*/ unsigned UsageTrackingAllocator<T,BUCKETS,MINSIZE,SCALE>::bucket_index(size_t size)
  {
    size_t index = 0;
    size_t bsize = MINSIZE;
    while((size > bsize) && (index < (BUCKETS-1))) {
      index++;
      bsize += (bsize * SCALE) >> 10;
    }
    return index;
  }

  template <typename T, size_t BUCKETS, size_t MINSIZE, size_t SCALE>
  void UsageTrackingAllocator<T,BUCKETS,MINSIZE,SCALE>::report(void)
  {
    printf("all sizes: ");
    global_stats.report();

    size_t min_size = 1;
    size_t max_size = MINSIZE;
    for(size_t i = 0; i < BUCKETS; i++) {
      if(buckets[i].total_allocs == 0)
	continue;

      if(i == (BUCKETS - 1))
	printf("%5zd +       B: ", min_size);
      else
	printf("%5zd - %5zd B: ", min_size, max_size);

      buckets[i].report();
      min_size = max_size + 1;
      max_size += (max_size * SCALE) >> 10;
    }
  }

  template <typename T, size_t BUCKETS, size_t MINSIZE, size_t SCALE>
  UsageTrackingAllocator<T,BUCKETS,MINSIZE,SCALE>::BucketInfo::BucketInfo(void)
    : cur_allocs(0), max_allocs(0), total_allocs(0)
    , cur_bytes(0), max_bytes(0), total_bytes(0)
    , sum_alloc_time(0), sum_alloc_time2(0), max_alloc_time(0)
    , sum_free_time(0), sum_free_time2(0), max_free_time(0)
  {}

  template <typename T>
  void atomic_max(T *addr, T newval)
  {
    if(newval > *addr) {
      T oldval = __sync_fetch_and_add(addr, 0);
      while(oldval < newval) {
	T expval = oldval;
	oldval = __sync_val_compare_and_swap(addr, oldval, newval);
	if(oldval == expval) break;
      }
    }
  }

  template <typename T, size_t BUCKETS, size_t MINSIZE, size_t SCALE>
  void UsageTrackingAllocator<T,BUCKETS,MINSIZE,SCALE>::BucketInfo::record_alloc(size_t bytes, long long nanoseconds)
  {
    size_t new_allocs = __sync_add_and_fetch(&cur_allocs, 1);
    atomic_max(&max_allocs, new_allocs);
    __sync_fetch_and_add(&total_allocs, 1);

    size_t new_bytes = __sync_add_and_fetch(&cur_bytes, bytes);
    atomic_max(&max_bytes, new_bytes);
    __sync_fetch_and_add(&total_bytes, bytes);

    if(nanoseconds > 0) {
      __sync_fetch_and_add(&sum_alloc_time, nanoseconds);
      __sync_fetch_and_add(&sum_alloc_time2, nanoseconds * nanoseconds);
      atomic_max(&max_alloc_time, nanoseconds);
    }
  }

  template <typename T, size_t BUCKETS, size_t MINSIZE, size_t SCALE>
  void UsageTrackingAllocator<T,BUCKETS,MINSIZE,SCALE>::BucketInfo::record_free(size_t bytes, long long nanoseconds)
  {
    __sync_fetch_and_sub(&cur_allocs, 1);
    __sync_fetch_and_sub(&cur_bytes, bytes);

    if(nanoseconds > 0) {
      __sync_fetch_and_add(&sum_free_time, nanoseconds);
      __sync_fetch_and_add(&sum_free_time2, nanoseconds * nanoseconds);
      atomic_max(&max_free_time, nanoseconds);
    }
  }

  template <typename T, size_t BUCKETS, size_t MINSIZE, size_t SCALE>
  void UsageTrackingAllocator<T,BUCKETS,MINSIZE,SCALE>::BucketInfo::report(void)
  {
    size_t scaled_total, scaled_peak;
    char units_total, units_peak;
    if(total_bytes < 100000) {
      scaled_total = total_bytes;
      units_total = ' ';
    } else if(total_bytes < 102400000ULL) {
      scaled_total = total_bytes >> 10;
      units_total = 'k';
    } else {
      scaled_total = total_bytes >> 20;
      units_total = 'M';
    }
    if(max_bytes < 100000) {
      scaled_peak = max_bytes;
      units_peak = ' ';
    } else if(max_bytes < 102400000ULL) {
      scaled_peak = max_bytes >> 10;
      units_peak = 'k';
    } else {
      scaled_peak = max_bytes >> 20;
      units_peak = 'M';
    }
    printf("total = %6zd (%5zd%cB), peak = %6zd (%5zd%cB)\n",
	   total_allocs, scaled_total, units_total,
	   max_allocs, scaled_peak, units_peak);
    if(total_allocs > 0) {
      long long avg_alloc_time = sum_alloc_time / total_allocs;
      long long stddev_alloc_time = sqrt((1.0 * total_allocs * sum_alloc_time2 - sum_alloc_time * sum_alloc_time)) / total_allocs;
      printf("            allocs: avg = %4lld ns, dev = %3lld ns, long = %7lld ns\n",
	     avg_alloc_time,
	     stddev_alloc_time,
	     max_alloc_time);
    }
    size_t total_frees = total_allocs - cur_allocs;
    if(total_frees > 0) {
      long long avg_free_time = sum_free_time / total_frees;
      long long stddev_free_time = sqrt((1.0 * total_frees * sum_free_time2 - sum_free_time * sum_free_time)) / total_frees;
      printf("             frees: avg = %4lld ns, dev = %3lld ns, long = %7lld ns\n",
	     avg_free_time,
	     stddev_free_time,
	     max_free_time);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class DefaultAllocator

  namespace ThreadLocal {
    /*__thread*/ Allocator *my_allocator = 0;
  };

  template <typename T>
  class DefaultAllocator {
  public:
    DefaultAllocator(void);
    virtual ~DefaultAllocator(void);

    T *alloc;
    ShareableMemory mapping;
    typedef UsageTrackingAllocator<BumpAllocator, 16> BUMPALLOC;
    BUMPALLOC *bump;
  };

  template <typename T>
  DefaultAllocator<T>::DefaultAllocator(void)
  {
    alloc = 0;//new T;
    //ThreadLocal::my_allocator = alloc;
    // HACK: claim all of memory
    //static_ranges.default_allocator = alloc;
    //Allocator::register_memory_range(alloc, 0, size_t(-1));
    ranges = &static_ranges;

    mapping.size = 1 << 30;
    bool ok = mapping.map();
    assert(ok);
    bump = new(mapping.base) BUMPALLOC;
    // subtract space for the metadata
    bump->set_pool_location(bump + 1, (1 << 30) - sizeof(BUMPALLOC));
    Allocator::register_memory_range(bump, mapping.base, 1 << 30);
    ThreadLocal::my_allocator = bump;
  }

  template <typename T>
  DefaultAllocator<T>::~DefaultAllocator(void)
  {
    //printf("libc allocator:\n");
    //alloc->report();
    printf("bump allocator:\n");
    bump->report();
    //ThreadLocal::my_allocator = 0;
  }

  //DefaultAllocator<LibcMalloc> libc_allocator;
  UsageTrackingAllocator<LibcMalloc, 16> libc_allocator;

  /*static*/ Allocator *Allocator::libc_allocator(void)
  {
    return &Realm::libc_allocator;
  }

  ShareableMemory shared_bump_mapping;
  typedef UsageTrackingAllocator<BumpAllocator, 16> BUMPALLOC;
  static BUMPALLOC *bump = 0;

  static void report_bump_stats(void)
  {
    bump->report();
  }

  void create_shared_bump_allocator(size_t bytes)
  {
    printf("creating shared allocator for isolated processors...\n");
    ranges = &static_ranges;

    shared_bump_mapping.size = 1 << 30;
    bool ok = shared_bump_mapping.map();
    assert(ok);
    bump = new(shared_bump_mapping.base) BUMPALLOC;
    // subtract space for the metadata
    bump->set_pool_location(bump + 1, (1 << 30) - sizeof(BUMPALLOC));
    Allocator::register_memory_range(bump, shared_bump_mapping.base, 1 << 30);
    ThreadLocal::my_allocator = bump;
    atexit(report_bump_stats);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ShareableMemory

  ShareableMemory::ShareableMemory(void)
    : is_active(false)
    , base(0)
    , size(0)
  {}

  ShareableMemory::~ShareableMemory(void)
  {
    // TODO: unmap memory?
  }

  bool ShareableMemory::map(void)
  {
    assert(!is_active);
    assert(size > 0);

    int prot = ((((default_access & CAN_READ) != 0) ? PROT_READ : 0) |
		(((default_access & CAN_WRITE) != 0) ? PROT_WRITE : 0) |
		(((default_access & CAN_EXEC) != 0) ? PROT_EXEC : 0));

    int flags = MAP_ANONYMOUS | MAP_SHARED;
    if(!reserve_memory)
      flags |= MAP_NORESERVE;

    void *mmap_base = mmap(0, size, prot, flags, -1 /*fd*/, 0);
    if(reinterpret_cast<intptr_t>(mmap_base) == -1)
      return false;

    if(numa_domain != -1) {
      if(!numasysif_bind_mem(numa_domain, mmap_base, size, false)) {
	munmap(mmap_base, size);
	return false;
      }
    }

    if(pin_memory) {
      int ret = mlock(mmap_base, size);
      if(ret != 0) {
	munmap(mmap_base, size);
	return false;
      }
    }

    base = mmap_base;
    is_active = true;
    return true;
  }

  bool ShareableMemory::unmap(void)
  {
    assert(is_active);

    // probably don't need to unlock things explicitly
    if(pin_memory) {
      int ret = munlock(base, size);
      NONMALLOC_ASSERT(ret == 0);
    }

    int ret = munmap(base, size);
    NONMALLOC_ASSERT(ret == 0);

    base = 0;
    is_active = false;
    return true;
  }

  void discard_data(size_t offset, size_t bytes)
  {
    // ignored for now
  }

};

#ifdef REALM_HIJACK_MALLOC
// hijack standard malloc and friends
extern "C" {
  void *malloc(size_t size)
  {
    Realm::Allocator *allocator = Realm::ThreadLocal::my_allocator;
    void *newptr = (allocator ? allocator->malloc(size, 0) :
		                __libc_malloc(size));
#ifdef SANITY_CHECK_ALLOCATIONS_IN_RANGE
    NONMALLOC_ASSERT(Realm::Allocator::find_allocator(newptr) == allocator);
#endif
    return newptr;
  }

  void *calloc(size_t nmemb, size_t size)
  {
    size_t bytes = nmemb * size;
    if(bytes > 0) {
      Realm::Allocator *allocator = Realm::ThreadLocal::my_allocator;
      void *newptr = (allocator ? allocator->malloc(bytes, 0) :
                                  __libc_malloc(bytes));
#ifdef SANITY_CHECK_ALLOCATIONS_IN_RANGE
      NONMALLOC_ASSERT(Realm::Allocator::find_allocator(newptr) == allocator);
#endif
      memset(newptr, 0, bytes);
      return newptr;
    } else
      return 0;    
  }

  // three different ways people can ask for aligned allocs
  int posix_memalign(void **memptr, size_t alignment, size_t size)
  {
    if(size > 0) {
      Realm::Allocator *allocator = Realm::ThreadLocal::my_allocator;
      if(allocator)
	*memptr = allocator->malloc(size, alignment);
      else
	*memptr = __libc_memalign(alignment, size);
#ifdef SANITY_CHECK_ALLOCATIONS_IN_RANGE
      if(*memptr != 0)
	NONMALLOC_ASSERT(Realm::Allocator::find_allocator(*memptr) == allocator);
#endif
      return (*memptr != 0) ? 0 : ENOMEM;
    } else {
      *memptr = 0;
      return 0;
    }
  }

  void *aligned_alloc(size_t alignment, size_t size)
  {
    Realm::Allocator *allocator = Realm::ThreadLocal::my_allocator;
    void *newptr = (allocator ? allocator->malloc(size, alignment) :
                                __libc_memalign(alignment, size));
#ifdef SANITY_CHECK_ALLOCATIONS_IN_RANGE
    NONMALLOC_ASSERT(Realm::Allocator::find_allocator(newptr) == allocator);
#endif
    return newptr;
  }

  void *memalign(size_t alignment, size_t size)
  {
    Realm::Allocator *allocator = Realm::ThreadLocal::my_allocator;
    void *newptr = (allocator ? allocator->malloc(size, alignment) :
                                __libc_memalign(alignment, size));
#ifdef SANITY_CHECK_ALLOCATIONS_IN_RANGE
    NONMALLOC_ASSERT(Realm::Allocator::find_allocator(newptr) == allocator);
#endif
    return newptr;
  }

  // realloc and free are different because they make sure the request is
  //  sent to the same allocator that performed the original allocation
  void *realloc(void *ptr, size_t size)
  {
    // realloc with a null old pointer is basically a malloc - use that
    //  path since there is no original allocator in this case
    if(!ptr)
      return malloc(size);

    Realm::Allocator *allocator = Realm::Allocator::find_allocator(ptr);
    void *newptr = (allocator ? allocator->realloc(ptr, size) :
		                __libc_realloc(ptr, size));
#ifdef SANITY_CHECK_ALLOCATIONS_IN_RANGE
    NONMALLOC_ASSERT(Realm::Allocator::find_allocator(newptr) == allocator);
#endif
    return newptr;
  }

  void free(void *ptr)
  {
    // free with a null pointer is a nop
    if(!ptr)
      return;

    Realm::Allocator *allocator = Realm::Allocator::find_allocator(ptr);
    if(allocator)
      allocator->free(ptr);
    else
      __libc_free(ptr);
  }
};
#endif

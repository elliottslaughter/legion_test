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

#include <sys/types.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <stdio.h>

extern "C" {
  void *__libc_malloc(size_t size);
  void *__libc_memalign(size_t alignment, size_t size);
  void *__libc_realloc(void *ptr, size_t size);
  void __libc_free(void *ptr);
};

namespace Realm {

  class Allocator {
  public:
    virtual ~Allocator(void);
    virtual void *malloc(size_t size, size_t alignment) = 0;
    virtual void *realloc(void *oldptr, size_t new_size) = 0;
    virtual void free(void *ptr) = 0;
  };

  Allocator::~Allocator(void)
  {}

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
  
  class BumpAllocator : public Allocator {
  public:
    BumpAllocator(size_t size);
    virtual ~BumpAllocator(void);

    virtual void *malloc(size_t size, size_t alignment);
    virtual void *realloc(void *oldptr, size_t new_size);
    virtual void free(void *ptr);

    // this descriptor needs to be in shared memory too!
    struct PoolDescriptor {
      intptr_t pool_start;
      intptr_t pool_pos;
      intptr_t pool_end;
    };
    PoolDescriptor *pool;
    size_t pool_size;
  };

  BumpAllocator::BumpAllocator(size_t size)
    : pool_size(size)
  {
    void *mmap_base = mmap(0, pool_size,
			   PROT_READ | PROT_WRITE,
			   MAP_ANONYMOUS | MAP_SHARED | MAP_NORESERVE,
			   -1 /*fd*/, 0);
    assert(mmap_base != reinterpret_cast<void *>(-1));
    pool = reinterpret_cast<PoolDescriptor *>(mmap_base);
    pool->pool_start = reinterpret_cast<intptr_t>(mmap_base);
    pool->pool_pos = pool->pool_start + sizeof(PoolDescriptor);
    pool->pool_end = pool->pool_start + pool_size;
  }

  BumpAllocator::~BumpAllocator(void)
  {
    int ret = munmap(pool, pool_size);
    assert(ret == 0);
  }

  void *BumpAllocator::malloc(size_t size, size_t alignment)
  {
    while(true) {
      intptr_t cur_pos = __sync_fetch_and_add(&pool->pool_pos, 0);

      intptr_t base = cur_pos + sizeof(size_t);
      if(alignment > 1) {
	intptr_t leftover = base % alignment;
	if(leftover > 0) {
	  assert((alignment % sizeof(size_t)) == 0);
	  base += alignment - leftover;
	}
      }
      (reinterpret_cast<size_t *>(base))[-1] = size;
      intptr_t new_pos = base + size;
      // align to sizeof(size_t) if needed
      {
	intptr_t leftover = size % sizeof(size_t);
	if(leftover > 0)
	  new_pos += sizeof(size_t) - leftover;
      }
      assert(new_pos <= pool->pool_end);

      // compare-and-swap to actually claim memory - loop around on failure
      if(__sync_bool_compare_and_swap(&pool->pool_pos, cur_pos, new_pos))
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

  template <typename T, size_t BUCKETS, size_t MINSIZE = 8, size_t SCALE=1024>
  class UsageTrackingAllocator : public T {
  public:
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

      void record_alloc(size_t bytes);
      void record_free(size_t bytes);

      size_t cur_allocs, max_allocs, total_allocs;
      size_t cur_bytes, max_bytes, total_bytes;
    };

    BucketInfo buckets[BUCKETS], global_stats;
  };

  template <typename T, size_t BUCKETS, size_t MINSIZE, size_t SCALE>
  void *UsageTrackingAllocator<T,BUCKETS,MINSIZE,SCALE>::malloc(size_t size, size_t alignment)
  {
    // we need to squirrel away some space to keep the size info
    size_t pad = sizeof(AllocInfo);
    if((alignment > 0) && ((pad % alignment) != 0))
      pad = ((pad / alignment) + 1) * alignment;

    void *ptr = T::malloc(size + pad, alignment);
    if(!ptr) return 0;

    ptr = reinterpret_cast<char *>(ptr) + pad;
    AllocInfo *info = reinterpret_cast<AllocInfo *>(ptr) - 1;
    info->magic = AllocInfo::MAGIC_VALUE;
    info->size = size;
    info->pad = pad;
    printf("alloc ptr=%p size=%zd pad=%zd\n", ptr, size, pad);

    unsigned idx = bucket_index(size);
    buckets[idx].record_alloc(size);
    global_stats.record_alloc(size);
    return ptr;
  }

  template <typename T, size_t BUCKETS, size_t MINSIZE, size_t SCALE>
  void *UsageTrackingAllocator<T,BUCKETS,MINSIZE,SCALE>::realloc(void *oldptr, size_t new_size)
  {
    if(oldptr != 0) {
      // get the old size and also check the magic value
      const AllocInfo *info = reinterpret_cast<AllocInfo *>(oldptr) - 1;
      if(info->magic != AllocInfo::MAGIC_VALUE) {
	printf("%p is not our alloc?\n", oldptr);
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
      printf("%p is not our alloc?\n", ptr);
      T::free(ptr);
      return;
    }

    printf("free %p size=%zd pad=%d\n", ptr, info->size, info->pad);

    unsigned idx = bucket_index(info->size);
    buckets[idx].record_free(info->size);
    global_stats.record_free(info->size);    

    // move pointer back down to base of underlying alloc for free call
    ptr = reinterpret_cast<char *>(ptr) - info->pad;
    T::free(ptr);
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
    printf("all sizes: %zd/%zd total, %zd/%zd peak\n",
	   global_stats.total_allocs, global_stats.total_bytes,
	   global_stats.max_allocs, global_stats.max_bytes);
    size_t min_size = 1;
    size_t max_size = MINSIZE;
    for(size_t i = 0; i < BUCKETS; i++) {
      printf("%zd - %zd B: %zd/%zd total, %zd/%zd peak\n",
	     min_size, max_size,
	     buckets[i].total_allocs, buckets[i].total_bytes,
	     buckets[i].max_allocs, buckets[i].max_bytes);
      min_size = max_size + 1;
      max_size += (max_size * SCALE) >> 10;
    }
  }

  template <typename T, size_t BUCKETS, size_t MINSIZE, size_t SCALE>
  UsageTrackingAllocator<T,BUCKETS,MINSIZE,SCALE>::BucketInfo::BucketInfo(void)
    : cur_allocs(0), max_allocs(0), total_allocs(0)
    , cur_bytes(0), max_bytes(0), total_bytes(0)
  {}

  template <typename T, size_t BUCKETS, size_t MINSIZE, size_t SCALE>
  void UsageTrackingAllocator<T,BUCKETS,MINSIZE,SCALE>::BucketInfo::record_alloc(size_t bytes)
  {
    size_t new_allocs = __sync_add_and_fetch(&cur_allocs, 1);
    if(new_allocs > max_allocs) {
      size_t old_allocs = max_allocs;
      while(new_allocs > old_allocs)
	old_allocs = __sync_val_compare_and_swap(&max_allocs, old_allocs, new_allocs);
    }
    __sync_fetch_and_add(&total_allocs, 1);

    size_t new_bytes = __sync_add_and_fetch(&cur_bytes, bytes);
    if(new_bytes > max_bytes) {
      size_t old_bytes = max_bytes;
      while(new_bytes > old_bytes)
	old_bytes = __sync_val_compare_and_swap(&max_bytes, old_bytes, new_bytes);
    }
    __sync_fetch_and_add(&total_bytes, bytes);
  }

  template <typename T, size_t BUCKETS, size_t MINSIZE, size_t SCALE>
  void UsageTrackingAllocator<T,BUCKETS,MINSIZE,SCALE>::BucketInfo::record_free(size_t bytes)
  {
    __sync_fetch_and_sub(&cur_allocs, 1);
    __sync_fetch_and_sub(&cur_bytes, bytes);
  }

  namespace ThreadLocal {
    /*__thread*/ Allocator *my_allocator = 0;
  };

  template <typename T>
  class DefaultAllocator {
  public:
    DefaultAllocator(void);
    virtual ~DefaultAllocator(void);

    T *alloc;
  };

  template <typename T>
  DefaultAllocator<T>::DefaultAllocator(void)
  {
    alloc = new T;
    ThreadLocal::my_allocator = alloc;
  }

  template <typename T>
  DefaultAllocator<T>::~DefaultAllocator(void)
  {
    alloc->report();
    //ThreadLocal::my_allocator = 0;
  }

  //DefaultAllocator<LibcMalloc> libc_allocator;
  DefaultAllocator<UsageTrackingAllocator<LibcMalloc, 16> > libc_allocator;

};

#ifdef REALM_HIJACK_MALLOC
// hijack standard malloc and friends
extern "C" {
  void *malloc(size_t size)
  {
    Realm::Allocator *allocator = Realm::ThreadLocal::my_allocator;
    if(allocator)
      return allocator->malloc(size, 0);
    else
      return __libc_malloc(size);
  }

  void *calloc(size_t nmemb, size_t size)
  {
    size_t bytes = nmemb * size;
    if(bytes > 0) {
      void *ptr;
      Realm::Allocator *allocator = Realm::ThreadLocal::my_allocator;
      if(allocator)
	ptr = allocator->malloc(bytes, 0);
      else
	ptr = __libc_malloc(bytes);
      memset(ptr, 0, bytes);
      return ptr;
    } else
      return 0;    
  }

  void *realloc(void *ptr, size_t size)
  {
    Realm::Allocator *allocator = Realm::ThreadLocal::my_allocator;
    if(allocator)
      return allocator->realloc(ptr, size);
    else
      return  __libc_realloc(ptr, size);
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
      return (*memptr != 0) ? 0 : ENOMEM;
    } else {
      *memptr = 0;
      return 0;
    }
  }

  void *aligned_alloc(size_t alignment, size_t size)
  {
    Realm::Allocator *allocator = Realm::ThreadLocal::my_allocator;
    if(allocator)
      return allocator->malloc(size, alignment);
    else
      return __libc_memalign(alignment, size);
  }

  void *memalign(size_t alignment, size_t size)
  {
    Realm::Allocator *allocator = Realm::ThreadLocal::my_allocator;
    if(allocator)
      return allocator->malloc(size, alignment);
    else
      return __libc_memalign(alignment, size);
  }

  void free(void *ptr)
  {
    Realm::Allocator *allocator = Realm::ThreadLocal::my_allocator;
    if(allocator)
      allocator->free(ptr);
    else
      __libc_free(ptr);
  }
};
#endif

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

extern "C" {
  void *__libc_malloc(size_t size);
  void *__libc_memalign(size_t alignment, size_t size);
  void *__libc_realloc(void *ptr, size_t size);
  void __libc_free(void *ptr);
};

namespace Realm {
  
  // this descriptor needs to be in shared memory too!
  struct PoolDescriptor {
    intptr_t pool_start;
    intptr_t pool_pos;
    intptr_t pool_end;
  };
  PoolDescriptor *pool = 0;

  void *do_malloc(size_t size, size_t alignment)
  {
    if(pool == 0) {
      // create pool
      size_t pool_size = 1 << 30; // 1 GB
      void *mmap_base = mmap(0, pool_size,
			     PROT_READ | PROT_WRITE,
			     MAP_ANONYMOUS | MAP_SHARED | MAP_NORESERVE,
			     -1 /*fd*/, 0);
      assert(mmap_base != reinterpret_cast<void *>(-1));
      PoolDescriptor *newpool = reinterpret_cast<PoolDescriptor *>(mmap_base);
      newpool->pool_start = reinterpret_cast<intptr_t>(mmap_base);
      newpool->pool_pos = newpool->pool_start + sizeof(PoolDescriptor);
      newpool->pool_end = newpool->pool_start + pool_size;
      pool = newpool;
    }

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

  void do_free(void *ptr)
  {
    // do nothing
  }

  void *do_realloc(void *oldptr, size_t new_size)
  {
    if(oldptr != 0) {
      // size of 0 is a free in disguise
      if(new_size == 0) {
	do_free(oldptr);
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
      void *newptr = do_malloc(new_size, 0);
      memcpy(newptr, oldptr, old_size);
      do_free(oldptr);
      return newptr;
    } else {
      if(new_size == 0)
	return 0;

      return do_malloc(new_size, 0);
    }
  }

#define USE_BUMP_ALLOC

  void *custom_malloc(size_t size, size_t alignment)
  {
#ifdef USE_BUMP_ALLOC
    return do_malloc(size, alignment);
#else
    if(size > 0) {
      if(alignment > 0)
	return __libc_memalign(alignment, size);
      else
	return __libc_malloc(size);
    } else
      return 0;
#endif
  }

  void *custom_realloc(void *ptr, size_t size)
  {
#ifdef USE_BUMP_ALLOC
    return do_realloc(ptr, size);
#else
    if(ptr != 0) {
      if(size > 0) {
	return __libc_realloc(ptr, size);
      } else {
	__libc_free(ptr);
	return 0;
      }
    } else {
      if(size > 0)
	return __libc_malloc(size);
      else
	return 0;
    }
#endif
  }

  void custom_free(void *ptr)
  {
#ifdef USE_BUMP_ALLOC
    do_free(ptr);
#else
    if(ptr)
      __libc_free(ptr);
#endif
  }

};

#ifdef REALM_HIJACK_MALLOC
// hijack standard malloc and friends
extern "C" {
  void *malloc(size_t size)
  {
    return Realm::custom_malloc(size, 0);
  }

  void *calloc(size_t nmemb, size_t size)
  {
    size_t bytes = nmemb * size;
    if(bytes > 0) {
      void *ptr = Realm::custom_malloc(bytes, 0);
      memset(ptr, 0, bytes);
      return ptr;
    } else
      return 0;    
  }

  void *realloc(void *ptr, size_t size)
  {
    return Realm::custom_realloc(ptr, size);
  }

  // three difference ways people can ask for aligned allocs
  int posix_memalign(void **memptr, size_t alignment, size_t size)
  {
    if(size > 0) {
      *memptr = Realm::custom_malloc(size, alignment);
      return (*memptr != 0) ? 0 : ENOMEM;
    } else {
      *memptr = 0;
      return 0;
    }
  }

  void *aligned_alloc(size_t alignment, size_t size)
  {
    return Realm::custom_malloc(size, alignment);
  }

  void *memalign(size_t alignment, size_t size)
  {
    return Realm::custom_malloc(size, alignment);
  }

  void free(void *ptr)
  {
    return Realm::custom_free(ptr);
  }
};
#endif

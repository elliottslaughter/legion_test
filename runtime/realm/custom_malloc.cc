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

extern "C" {
  void *__libc_malloc(size_t size);
  void *__libc_memalign(size_t alignment, size_t size);
  void *__libc_realloc(void *ptr, size_t size);
  void __libc_free(void *ptr);
};

namespace Realm {

  void *custom_malloc(size_t size, size_t alignment)
  {
    if(size > 0) {
      if(alignment > 0)
	return __libc_memalign(alignment, size);
      else
	return __libc_malloc(size);
    } else
      return 0;
  }

  void *custom_realloc(void *ptr, size_t size)
  {
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
  }

  void custom_free(void *ptr)
  {
    if(ptr)
      __libc_free(ptr);
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

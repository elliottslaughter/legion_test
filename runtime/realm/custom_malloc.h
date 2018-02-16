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

#ifndef REALM_CUSTOM_MALLOC_H
#define REALM_CUSTOM_MALLOC_H

#include "realm/realm_config.h"
#include "realm/utils.h"

#include <sys/types.h>

namespace Realm {

  // custom allocators must inherit from this interface

  class Allocator {
  public:
    virtual ~Allocator(void);
    virtual void *malloc(size_t size, size_t alignment) = 0;
    virtual void *realloc(void *oldptr, size_t new_size) = 0;
    virtual void free(void *ptr) = 0;

    // there's always an allocator that just falls through to the
    //  malloc/free implementation in libc
    static Allocator *libc_allocator(void);

    static Allocator *find_allocator(void *ptr);
    static void register_memory_range(Allocator *alloc, void *base, size_t bytes);
    static void unregister_memory_range(Allocator *alloc, void *base, size_t bytes);

    // returns the allocator currently assigned to perform allocations for
    //  the calling thread
    static Allocator *get_current_allocator(void);

    // sets the allocator currently assigned to perform allocations for
    //  the calling thread
    static void set_current_allocator(Allocator *alloc);
  };

  class ScopedAllocatorPush {
  public:
    ScopedAllocatorPush(Allocator *new_alloc)
      : old_alloc(Allocator::get_current_allocator())
    {
      Allocator::set_current_allocator(new_alloc);
    }

    ~ScopedAllocatorPush(void)
    {
      Allocator::set_current_allocator(old_alloc);
    }

  protected:
    Allocator *old_alloc;
  };

  class ShareableMemory {
  public:
    ShareableMemory(void);
    ~ShareableMemory(void);

    enum {
      CAN_READ = 1,
      CAN_WRITE = 2,
      CAN_READWRITE = 3,
      CAN_EXEC = 4,
    };

    WithDefault<int,  -1>            numa_domain;
    WithDefault<bool, true>          reserve_memory;
    WithDefault<bool, false>         pin_memory;
    WithDefault<int,  CAN_READWRITE> default_access;

    bool is_active;
    void *base;
    size_t size;

    bool map(void);
    bool unmap(void);

    void discard_data(size_t offset, size_t bytes);
  };

  void create_shared_bump_allocator(size_t bytes);

};

#endif

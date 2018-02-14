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

// little helper utilities for Realm code
// none of this is Realm-specific, but it's put in the Realm namespace to
//  reduce the chance of conflicts

#ifndef REALM_UTILS_H
#define REALM_UTILS_H

#include <vector>
#include <map>
#include <string>
#include <sstream>

namespace Realm {
    
  // helpers for deleting contents STL containers of pointers-to-things

  template <typename T>
  void delete_container_contents(std::vector<T *>& v, bool clear_cont = true)
  {
    for(typename std::vector<T *>::iterator it = v.begin();
	it != v.end();
	it++)
      delete (*it);

    if(clear_cont)
      v.clear();
  }

  template <typename K, typename V>
  void delete_container_contents(std::map<K, V *>& m, bool clear_cont = true)
  {
    for(typename std::map<K, V *>::iterator it = m.begin();
	it != m.end();
	it++)
      delete it->second;

    if(clear_cont)
      m.clear();
  }


  // helper class that lets you build a formatted std::string as a single expression:
  //  /*std::string s =*/ stringbuilder() << ... << ... << ...;

  class stringbuilder {
  public:
    operator std::string(void) const { return ss.str(); }
    template <typename T>
    stringbuilder& operator<<(T data) { ss << data; return *this; }
  protected:
    std::stringstream ss;
  };

  // little helper class that defines a default value for a member variable
  //  in the header rather than in the containing object's constructor
  //  implementation
  template <typename T, T _DEFAULT>
  struct WithDefault {
  public:
    static const T DEFAULT_VALUE = _DEFAULT;

    WithDefault(void) : val(_DEFAULT) {}
    WithDefault(T _val) : val(_val) {}

    operator T(void) const { return val; }
    WithDefault<T,_DEFAULT>& operator=(T newval) { val = newval; return *this; }

  protected:
    T val;
  };

}; // namespace Realm

#endif // ifndef REALM_UTILS_H

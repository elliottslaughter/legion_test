/* Copyright 2013 Stanford University and Los Alamos National Security, LLC
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

#include "foreign_cpp_main.lg.h"

void create_mappers(Machine *machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs) {}

int main(int argc, char **argv) {
  HighLevelRuntime::set_registration_callback(create_mappers);
  init_foreign_cpp_main_lg();
  HighLevelRuntime::set_top_level_task_id(TASK_TOPLEVEL);

  return HighLevelRuntime::start(argc, argv);
}

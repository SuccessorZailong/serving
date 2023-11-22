/* Copyright 2023 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_SERVING_SESSION_BUNDLE_SAVED_MODEL_CONFIG_H_
#define TENSORFLOW_SERVING_SESSION_BUNDLE_SAVED_MODEL_CONFIG_H_

#include <string>

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace serving {
namespace session_bundle {

Status MaybeLoadSavedModelConfig(const std::string& export_dir,
                                 SessionOptions* session_options);

}  // namespace session_bundle
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SESSION_BUNDLE_SAVED_MODEL_CONFIG_H_

#include <torch/extension.h>
#include "polygon_mask.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("polygon_mask", &PolygonMask);
}

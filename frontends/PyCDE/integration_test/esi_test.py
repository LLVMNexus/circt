# REQUIRES: esi-runtime, esi-cosim, rtl-sim
# RUN: rm -rf %t
# RUN: mkdir %t && cd %t
# RUN: %PYTHON% %s %t 2>&1
# RUN: esi-cosim.py -- %PYTHON% %S/test_software/esi_test.py cosim env

import pycde
from pycde import (AppID, Clock, Module, Reset, generator)
from pycde.bsp import cosim
from pycde.constructs import Wire
from pycde.esi import FuncService, MMIO
from pycde.types import (Bits, Channel, UInt)

import sys


class LoopbackInOutAdd7(Module):
  """Loopback the request from the host, adding 7 to the first 15 bits."""

  @generator
  def construct(ports):
    loopback = Wire(Channel(UInt(16)))
    args = FuncService.get_call_chans(AppID("loopback_add7"),
                                      arg_type=UInt(24),
                                      result=loopback)

    ready = Wire(Bits(1))
    data, valid = args.unwrap(ready)
    plus7 = data + 7
    data_chan, data_ready = loopback.type.wrap(plus7.as_uint(16), valid)
    ready.assign(data_ready)
    loopback.assign(data_chan)


class MMIOClient(Module):

  @generator
  def build(ports):
    mmio_read_bundle = MMIO.read(appid=AppID("mmio_client"))

    address_chan_wire = Wire(Channel(UInt(32)))
    address, address_valid = address_chan_wire.unwrap(1)
    response_data = (address + 7).as_bits(32)
    response_chan, response_ready = Channel(Bits(32)).wrap(
        response_data, address_valid)

    address_chan = mmio_read_bundle.unpack(data=response_chan)['offset']
    address_chan_wire.assign(address_chan)


class Top(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    LoopbackInOutAdd7()
    # MMIOClient()


if __name__ == "__main__":
  s = pycde.System(cosim.CosimBSP(Top),
                   name="ESILoopback",
                   output_directory=sys.argv[1])
  try:
    s.generate()
    s.run_passes(debug=True)
    s.compile()
    s.package()
    s.print()
  except Exception as e:
    s.print(file=open("error.mlir", "w"), enable_debug_info=True)
    raise e

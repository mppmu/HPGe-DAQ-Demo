using StruckVMEDevices.MemRegisters
using StruckVMEDevices.SIS3316Digitizers
using Observables
using TOML
using ProgressMeter
using LegendHDF5IO, HDF5

# ENV["JULIA_DEBUG"] = "StruckVMEDevices"

config = TOML.parsefile("daq-config.toml")

adc_hostname = config["adc"]

adc = SIS3316Digitizer(adc_hostname)

adc.readout_config = ReadoutConfig([1], 0.5, 2.0)

events = empty(adc.events[])

function process_events(new_events)
    # println("Events per s: $(adc.evtrate[])")
    append!(events, new_events)
    println("Captured $(length(new_events)) new events, $(length(events)) events in total.")
    nothing
end

on(process_events, adc.events)
set_capture!(adc, true)
@showprogress for i in 1:20#*60
    sleep(1)
end
set_capture!(adc, false)
off(adc.events, process_events)

HDF5.h5open("adcdata.lh5", "w") do f
    LegendHDF5IO.writedata(f, "raw", events)
end

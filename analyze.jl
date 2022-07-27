using LegendHDF5IO, HDF5
using ArraysOfArrays
using Statistics, StatsBase
using DSP, RadiationDetectorDSP
using RadiationSpectra
using Plots

events = HDF5.h5open("adcdata.lh5") do f
    LegendHDF5IO.readdata(f, "raw")
end

wfs = ArrayOfSimilarArrays(deepmap(Float32, events.samples))

plot(wfs[1:10])

wf = wfs[1]
wf_blcorr = wf .- mean(wf[1:1500])

deconv_filter = RadiationDetectorDSP.inv_cr_filter(13000)
deconv_wf = filt(deconv_filter, wf_blcorr)

plot(wf_blcorr)
plot!(deconv_wf)

eflt_wf = charge_trapflt!(copy(deconv_wf), 1000, 500)

plot(eflt_wf)
maximum(eflt_wf)


function energy_reco(wf)
    wf_blcorr = wf .- mean(wf[1:1500])
    deconv_filter = RadiationDetectorDSP.inv_cr_filter(13000)
    deconv_wf = filt(deconv_filter, wf_blcorr)
    eflt_wf = charge_trapflt!(copy(deconv_wf), 1000, 500)
    maximum(eflt_wf)
end

E_rec = energy_reco.(wfs)


h_uncal = fit(Histogram, E_rec, 0:1:2000)
plot(h_uncal, lt = :stepbins, yscale = :log10)

_, peakpos = RadiationSpectra.peakfinder(h_uncal)
vline!(peakpos)

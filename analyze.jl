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

deconv_filter = RadiationDetectorDSP.inv_cr_filter(14000)
deconv_wf = filt(deconv_filter, wf_blcorr)

plot(wf_blcorr)
plot!(deconv_wf)


function apply_inv_cr_flt_parallel!(A::AbstractMatrix{<:Real}, tau::Real, acc::AbstractVector{<:Real})
    T = eltype(A)
    alpha = 1 / (1 + T(tau))

    @assert axes(A,2) == axes(acc,1)

    acc .= 0

    @inbounds for i in axes(A,1) # loop over samples
        @simd for j in axes(A,2) # loop over waveforms
            x = A[i,j]
            y = x + acc[j]
            acc[j] = muladd(x, alpha, acc[j])
            A[i,j] = y
        end
    end
    return A
end


function apply_inv_cr_flt_parallel_tp!(A_tp::AbstractMatrix{<:Real}, tau::Real, acc::AbstractVector{<:Real})
    T = eltype(A_tp)
    alpha = 1 / (1 + T(tau))

    @assert axes(A_tp,1) == axes(acc,1)

    acc .= 0

    Base.Threads.@threads for j in axes(A_tp,2) # loop over samples
        @inbounds @simd for i in axes(A_tp,1) # loop over waveforms
            x = A_tp[i,j]
            y = x + acc[i]
            acc[i] = muladd(x, alpha, acc[i])
            A_tp[i,j] = y
        end
    end
    return A_tp
end


#=
acc = similar(flatview(wfs), length(wfs))
@time apply_inv_cr_flt_parallel_tp!(A_tp, 14000, acc);
=#


function apply_inv_cr_flt!(wf::AbstractVector{<:Real}, tau::Real)
    T = eltype(wf)
    alpha = 1 / (1 + T(tau))
    acc::T = 0

    for i in eachindex(wf)
        x = wf[i]
        y = x + acc
        acc = muladd(x, alpha, acc)
        wf[i] = y
    end

    return wf
end

deconv_wf = apply_inv_cr_flt!(copy(wf_blcorr), 14000)
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

E_cal = 1460/769.59 .* E_rec
h_cal = fit(Histogram, E_cal, 0:1:2700)


plot(h_cal, lt = :stepbins, yscale = :log10, xrange = (1450, 1470))

calibration_values = Dict(
    "annihilation" => 511.00,
    "Pb-207" => 569.70, 
    "Tl-208" => 583.19, 
    "Bi-214" => 609.31,
    "Cs-137" => 661.66, 
    "Ac-228" => 911.20,
    "Pb-207" => 1063.66,
    "Bi-214" => 1120.28,
    "Fe-56" => 1238.28,
    "K-40" => 1460.83,
    #1729.595, 
    "Bi-214" => 1764.49,
    # 2204.21, 2447.86
    "Tl-208/Pb-208" => 2614.53)



######
#calibration

photon_lines = sort(collect(values(calibration_values)))
h_cal, h_deconv, peakPositions, threshold, c, c_precal = RadiationSpectra.calibrate_spectrum(h_uncal, photon_lines)

p_uncal = plot(h_uncal, st=:step, label="Uncalibrated spectrum");
p_deconv = plot(h_deconv, st=:step, label = "Deconvoluted spectrum");
hline!([threshold], label = "threshold", lw = 1.5);
p_cal = plot(h_cal, st=:step, label="Calibrated spectrum", xlabel="E / keV");
vline!(photon_lines, lw=0.5, color=:red, label="Photon lines");
plot(p_uncal, p_deconv, p_cal, size=(800,700), layout=(3, 1))


E_cal = c .* E_rec
h_cal = fit(Histogram, E_cal, 0:0.5:2700)


identified = Dict(
    "annihilation" => 511.00,
    "Pb-207-1exc0" => 569.70, 
    "Tl-208" => 583.19, 
    "Bi-214-1" => 609.31,
    "Cs-137" => 661.66, 
    "Ac-228" => 911.20,
    "Pb-207-m" => 1063.66,
    "Bi-214-2" => 1120.28,
    "Fe-56" => 1238.28,
    "K-40" => 1460.83,
    "1" => 1729.595, 
    "Bi-214-3" => 1764.49,
    "Bi-214-4" => 2204.21, 
    "x3" =>2447.86,
    "Tl-208/Pb-208" => 2614.53,
    ##neu
    "Ac-228-2"=> 968.97,
    "Tl-208-sep" => 2614.53 - 511,
    "Tl-208-dep" => 2614.53 - 2*511,
    "Bi-214-5" => 768.36,
    "Bi-214-6" => 1377.669,
    "Bi-214-7" => 934.056,
    "Pb-214-1" => 351.93,
    "Pb-214-2" => 242.00,
    "Pb-214-3" => 295.22,
)


plot(h_cal, lt = :stepbins, yscale = :log10, size=(1200,800))
vline!(collect(values(identified)))


using Distributions, ValueShapes, BAT, MeasureBase, DensityInterface

expected(xs, v) = product_distribution(Poisson.(v.a .* pdf.(Ref(Normal(v.mu, v.sigma)), xs) .+ v.offs))

prior = NamedTupleDist(
    mu = Normal(1450, 20),
    sigma = Uniform(1, 10),
    a = Uniform(0, 5000),
    offs = Uniform(0, 1000)
)



likelihood = Likelihood(Base.Fix1(expected, xs), h.weights)

posterior = PosteriorMeasure(likelihood, prior)

r = bat_sample(posterior)

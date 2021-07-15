using Gen
@gen function foo(i::Int64)
    hr = @trace(uniform(0., 1.), :hit_rate)
    fp = @trace(uniform(0., 1.), :false_alarm)
    return (hr, fp)
end
@gen static function bar(k::Int64)
    args = collect(Int64, 1:k)
    rates = @trace(Gen.Map(foo)(args), :rates)
    hr, fp = rates[1]
    obs1 = @trace(uniform(0.0, hr), :obs1)
    obs2 = @trace(uniform(0.0, fp), :obs2)
    return nothing
end
@load_generated_functions
trace,_ = Gen.generate(bar, (3,))
choices = get_choices(trace)
display(choices)
trace, accepted = hmc(trace, select(:rates => 1 => :hit_rate,
                                   :rates => 1 => :false_alarm))

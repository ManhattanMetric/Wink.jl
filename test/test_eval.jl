@testset "evalengine" begin
    r = Wink.eval_in_main("1 + 1")
    @test r.ok
    @test r.value_repr == "2"
    @test isempty(r.output)

    r = Wink.eval_in_main("println(\"hey there\"); 5")
    @test r.ok
    @test occursin("hey there", r.output)
    @test r.value_repr == "5"

    # REPL soft-scope: assigning a global inside a top-level for loop works
    r = Wink.eval_in_main("wink_total = 0\nfor i in 1:3\n    wink_total += i\nend\nwink_total")
    @test r.ok
    @test r.value_repr == "6"

    # a parse error anywhere means nothing executes
    r = Wink.eval_in_main("wink_never_defined = 1\nthis is not julia ((")
    @test !r.ok
    @test occursin("parse error", r.error_text)
    @test !isdefined(Main, :wink_never_defined)

    r = Wink.eval_in_main("error(\"boom-boom\")")
    @test !r.ok
    @test occursin("boom-boom", r.error_text)

    # define-then-call in one snippet (world age across top-level exprs)
    r = Wink.eval_in_main("wink_eval_test_f(x) = 3x\nwink_eval_test_f(4)")
    @test r.ok
    @test r.value_repr == "12"

    r = Wink.eval_in_main("nothing")
    @test r.ok
    @test r.value_repr == "nothing"

    s = Wink.format_for_model(Wink.EvalResult(true, "VAL", "OUT", ""))
    @test occursin("value: VAL", s)
    @test occursin("OUT", s)
    s = Wink.format_for_model(Wink.EvalResult(false, "", "", "ERROR: nope"))
    @test occursin("ERROR: nope", s)
    @test !occursin("value:", s)
end

@testset "eval gate" begin
    old_confirm = Wink.CONFIG.confirm
    old_auto = Wink.CONFIG.autoeval
    try
        Wink.CONFIG.autoeval = false
        Wink.CONFIG.confirm = (k, t) -> false
        @test Wink.tool_eval_code("1+1") == Wink.DECLINED_MSG

        seen = Ref{Any}(nothing)
        Wink.CONFIG.confirm = (k, t) -> (seen[] = (k, t); true)
        @test occursin("value: 2", Wink.tool_eval_code("1+1"))
        @test seen[] == (:eval, "1+1")

        Wink.CONFIG.confirm = (k, t) -> error("must not be called")
        Wink.CONFIG.autoeval = true
        @test occursin("value: 2", Wink.tool_eval_code("1+1"))
    finally
        Wink.CONFIG.confirm = old_confirm
        Wink.CONFIG.autoeval = old_auto
    end
end

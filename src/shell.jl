# Shell execution as a first-class tool.
#
# Models — small local ones especially — reliably mangle Julia's Cmd API when
# asked to shell out from eval_code: they reach for Python/C idioms (`system`,
# `run("string")`, `run([...])`) that don't exist in Julia, or write a bare
# backtick literal that constructs a Cmd without ever running it. A dedicated
# tool that takes the command as one plain string matches what every model has
# seen a thousand times and sidesteps Cmd syntax entirely.

"""
    run_shell_command(command::AbstractString; dir = pwd()) -> (; exitcode, output)

Run `command` through the system shell (`sh -c`, or `cmd /c` on Windows) in
directory `dir`, with stdout and stderr merged in order into `output`. A
nonzero exit status is reported in `exitcode`, never thrown; Ctrl-C kills the
child process and propagates.
"""
function run_shell_command(command::AbstractString; dir::AbstractString = pwd())
    shell = Sys.iswindows() ? `cmd /c $command` : `sh -c $command`
    out = Pipe()
    proc = run(pipeline(Cmd(shell; dir = String(dir), ignorestatus = true);
            stdout = out, stderr = out, stdin = devnull); wait = false)
    close(out.in)
    output = try
        s = read(out, String)
        wait(proc)
        s
    catch e
        e isa InterruptException && kill(proc)
        rethrow()
    end
    return (; exitcode = proc.exitcode, output)
end

function format_shell_result(exitcode::Integer, output::AbstractString)
    body = isempty(strip(output)) ? "(no output)" : truncate_output(output)
    return exitcode == 0 ? body : "exit code: $exitcode\n$body"
end

"""
    run_shell(command::String, dir::String) -> String

Run a shell command and return its combined stdout/stderr (plus the exit code
when nonzero). This is the right tool for anything you would type in a
terminal — git, mkdir/ls/cp, curl, build tools, package managers — use it
instead of translating shell work into Julia code for eval_code. `command` is
one plain string passed to `sh -c`, so pipes, `&&`, redirection, quoting, and
globs all work. Each call runs in a fresh shell process: `cd` and environment
changes do not persist to the next call — to run somewhere specific, pass that
directory as `dir` (`""` runs in the session's current directory). Unless
auto-approval is on, the user is shown the command first and may decline; a
DECLINED result means do not retry it verbatim.
"""
tool_run_shell(command::String, dir::String) = tool_guard("run_shell") do
    isempty(strip(command)) && throw(WinkResolveError("command is empty"))
    d = strip(dir)
    isempty(d) || isdir(expanduser(d)) ||
        throw(WinkResolveError("directory not found: $d"))
    shown = isempty(d) ? command : "# in $d\n" * command
    CONFIG.autoeval || CONFIG.confirm(:shell, shown) || return DECLINED_MSG
    r = run_shell_command(command; dir = isempty(d) ? pwd() : expanduser(d))
    format_shell_result(r.exitcode, r.output)
end

push!(TOOL_SPECS, "run_shell" => tool_run_shell)

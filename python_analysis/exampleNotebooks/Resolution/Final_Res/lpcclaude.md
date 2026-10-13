# How we work on LPC / lxplus from a laptop

A guide for replicating our development setup. This describes *our* arrangement
(macOS + VS Code + Claude Code + Fermilab LPC and CERN lxplus), and flags the
places where you'd substitute something different on Linux or Windows.

_Last updated: 2026-08-20_

---

## 1. The mental model (read this first)

Everything below is one idea in three parts:

| Layer | Where it runs | What it's for |
|---|---|---|
| **Editor + AI agent** | your laptop | reading, writing, thinking, git-diff viewing |
| **Files** | the cluster, exposed on the laptop as a normal-looking folder via SSHFS | one source of truth, no copies to sync |
| **Execution** | the cluster, reached by `ssh` | builds, CMSSW, conda envs, condor, anything that needs `/cvmfs` |

So: **you edit remote files as if local, but you never *run* anything locally.**
There is exactly one copy of every file, it lives on the cluster, and the laptop
is a window onto it.

The reason we do it this way rather than the more obvious alternatives:

- **Not "clone locally and rsync"** — two copies means drift, and drift on an
  analysis repo eventually means a plot made from code that isn't the code you
  committed.
- **Not "VS Code Remote-SSH"** (which would be simpler and does work) — because
  the agent tooling we use runs on the laptop and can't be installed on the
  cluster. If you *don't* need laptop-side tooling, Remote-SSH is honestly the
  better editor experience and you should consider it; see §10.
- **Not "work in a terminal over ssh"** — you lose the editor, the notebook
  rendering, and the agent.

Everything complicated in this document is a consequence of that split, plus one
hard fact: **SSHFS is fine for reading, unreliable for bulk writing.**

---

## 2. Where the projects actually live

We have three storage areas, each mounted at a fixed path on the laptop:

| Laptop path | Remote path | Host | Use it for |
|---|---|---|---|
| `~/lpc/` | `/uscms_data/d3/<fnal-user>/` | `cmslpc-el9.fnal.gov` | CMS analysis (big scratch, visible from LPC and the EAF CMS image) |
| `~/nashome/` | `/nashome/<first-letter>/<fnal-user>/` | same LPC node | cross-facility work (visible from LPC, *all* EAF images, fnalu, gpvms) |
| `~/lxplus/` | `/afs/cern.ch/` (the whole cell) | `lxplus.cern.ch` | CERN-side work; AFS home is `~/lxplus/user/<l>/<cern-user>` |

Notes that will save you an afternoon:

- `~/lxplus/` is the **whole AFS cell**, not just your home. `~/lxplus/<X>` maps
  to `/afs/cern.ch/<X>`. That means a recursive `grep` from `~/lxplus/` would
  walk CERN-wide AFS over SSHFS. Don't.
- **EOS is not under any mount.** `/eos/user/...` at CERN and
  `/store/user/...` at FNAL are reached server-side (`ssh lxplus 'ls /eos/...'`)
  or over xrootd (`root://eosuser.cern.ch//eos/user/...`,
  `root://cmseos.fnal.gov//store/...`). At FNAL specifically, prefer xrootd over
  the `/eos/uscms` POSIX mount — that mount degrades the shared interactive node
  and isn't available on batch workers at all.
- Quotas differ wildly: AFS home ~10 GB, AFS work ≤100 GB, EOS ~1 TB, LPC
  `/uscms_data/d3` is the roomy one. Datasets go on EOS, not in a home area.
- Same file, different absolute path per facility: LPC sees `/uscms_data/d3/...`
  while the EAF CMS image sees `/uscms_data/data3/...`. **Don't hardcode either**
  in code that runs at both — use relative paths from a known cwd, branch on
  `socket.gethostname()`, or read a `$PROJECT_BASE` env var.

---

## 3. Prerequisites, by OS

### The SSHFS client

**macOS (what we run).** macFUSE needs a kernel extension, which is painful on
Apple Silicon. Use **FUSE-T** instead — it's userspace (implements a local NFS
server), no kext, and it survived a Sequoia→Tahoe major-version upgrade
untouched. Two signed `.pkg` installers:

- <https://github.com/macos-fuse-t/fuse-t/releases> → `fuse-t-*.pkg`
- <https://github.com/macos-fuse-t/sshfs/releases> → `fuse-t-sshfs-*.pkg`

Or with Homebrew: `brew tap macos-fuse-t/cask && brew install --cask fuse-t fuse-t-sshfs`.
Verify with `sshfs --version`.

**Linux.** Native and easier: `apt install sshfs` (Debian/Ubuntu) or
`dnf install fuse-sshfs` (Fedora/RHEL). Our mount script works nearly as-is —
drop the two macOS-only options (`-o volname=`, `-o noappledouble`) and replace
the `perl`-based timeout helper with plain `timeout(1)`. Linux inotify over
SSHFS still doesn't work, so keep the watcher exclusions in §7.

**Windows.** Two options. The one we'd recommend is **WSL2** — install a Linux
distro, do everything inside it exactly as the Linux instructions above, and
point VS Code at the WSL environment. The alternative is
[WinFsp](https://winfsp.dev/) + [sshfs-win](https://github.com/winfsp/sshfs-win),
which maps the remote path to a drive letter; it works but the OpenSSH
`ControlMaster` multiplexing we depend on (§4) is not available on native
Windows OpenSSH, so you'd re-authenticate constantly. WSL2 avoids that.

### Kerberos (Fermilab only)

You need `kinit`/`klist` for FNAL. macOS ships Heimdal at `/usr/bin/kinit`;
Linux wants `krb5-workstation` / `krb5-user`.

> **Trap we hit:** if you have conda or mamba active, bare `kinit`/`klist`
> resolve to *conda's* MIT Kerberos builds, which read a different credential
> cache (`KCM:`) than the one macOS `ssh` actually uses (`API:`). Result:
> `klist` swears you have no ticket while you do, and `kinit` stashes a fresh
> ticket somewhere `ssh` will never look. **Always call `/usr/bin/klist` and
> `/usr/bin/kinit` with the explicit path** in any script or check.

CERN needs no local Kerberos at all — see §5.

---

## 4. The SSH config — this is the load-bearing piece

Everything (the mount, every remote command, the Jupyter forward) rides on two
host aliases with **connection multiplexing**. `ControlMaster auto` means the
first connection authenticates and opens a socket; every later `ssh` reuses that
socket with no re-auth and no handshake latency. That's what makes
`ssh lpc '<cmd>'` cheap enough to use as the default way to run things.

Put this in `~/.ssh/config`:

```sshconfig
# Fermilab LPC. Kerberos ticket is created locally and delegated to the node.
Host lpc
    HostName cmslpc-el9.fnal.gov
    User <fnal-user>                 # omit if same as your laptop username
    GSSAPIAuthentication yes
    GSSAPIDelegateCredentials yes
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
    ServerAliveInterval 60
    ServerAliveCountMax 3
    LocalForward 18888 localhost:18888   # for the remote Jupyter kernel, see §9

# CERN lxplus. Auth is interactive password + 2FA, done ONCE, held open forever.
Host lxplus
    HostName lxplus.cern.ch
    User <cern-user>
    GSSAPIAuthentication yes
    GSSAPIDelegateCredentials yes
    PreferredAuthentications gssapi-with-mic,keyboard-interactive,password
    StrictHostKeyChecking accept-new
    LogLevel ERROR
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

The `ControlPersist` values differ deliberately, and the reason is the single
most important operational difference between the two facilities:

- **FNAL: `8h`.** You hold a Kerberos ticket locally (`kinit -f <user>@FNAL.GOV`,
  ~26 h lifetime) and `GSSAPIDelegateCredentials yes` forwards it. Re-establishing
  a dead master is free and silent — `ssh lpc true` just works. So an 8-hour
  idle expiry costs nothing.
- **CERN: `yes` (never expire).** CERN requires 2FA on **every new SSH
  connection from outside CERN** — `pam_2fa` runs even after a successful
  Kerberos/GSSAPI login, and CERN's KDC has no OTP pre-authentication. A local
  `kinit <user>@CERN.CH` therefore buys you nothing; it would only replace the
  password keystroke, never the 2FA prompt. **The only lever is keeping one
  master alive as long as possible**, so it must never time itself out — it
  should live until the TCP connection actually breaks.

  While that master lives, CERN's own hourly timer on the node
  (`cern-aklog-systemd-user`, running `kinit -R && aklog`) keeps your Kerberos
  ticket and AFS token fresh for up to **5 days** after login.

**Corollary for automation:** never try to script the CERN 2FA. Our tooling
attempts a `BatchMode=yes` connection (which either succeeds silently via an
existing credential or fails fast without prompting), and if that fails and no
terminal is attached, it prints instructions and stops.

---

## 5. Mounting

We keep four scripts in `~/lpc-tools/`. The core is `mount.sh`; the others are
thin wrappers that retarget it through environment variables:

```
mount.sh              # engine: LPC scratch by default
mount-nashome.sh      # LPC_REMOTE_PATH=/nashome/<l>/<user>, MOUNT_POINT=~/nashome
mount-lxplus.sh       # LPC_HOST=lxplus, path=/afs/cern.ch, AUTH_MODE=master
umount.sh             # + umount-nashome.sh, umount-lxplus.sh
```

The actual mount command, with every option chosen for a reason:

```bash
sshfs "lpc:/uscms_data/d3/<user>" "$HOME/lpc" \
    -o reconnect \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -o cache_timeout=60 \
    -o volname=lpc \        # macOS only — the name Finder shows
    -o noappledouble        # macOS only — stop ._* turd files
```

- `cache_timeout=60` caches stat/dir metadata locally for a minute. This is the
  single biggest speed knob: file browsing and `git status` stop round-tripping
  to Fermilab for every `lstat`. Your own writes invalidate their own cache
  entries immediately, so the only staleness is *server-side* changes (a fresh
  condor output, say) showing up up to 60 s late. Raise it if you can tolerate
  more lag.
- `reconnect` helps across brief network blips. It is **not** bulletproof across
  a long laptop sleep — expect to remount.
- The lxplus wrapper adds `-o transform_symlinks`, because AFS is full of
  absolute symlinks into `/afs/cern.ch/...`; this rewrites the ones pointing
  inside the mounted tree so they resolve through `~/lxplus`. Links to *other*
  AFS cells still break locally and must be resolved server-side.

### What the script does beyond calling sshfs

Roughly 100 of `mount.sh`'s lines are failure handling, and they're the part
worth copying:

1. **Detect a wedged mount.** A FUSE-T mount can stay listed in `mount` after
   its server dies, with every read hanging *uninterruptibly*. So before
   anything, probe with a hard-timeout `ls` (twice, to avoid false-positiving a
   merely-slow-but-alive mount and force-unmounting it out from under pending
   writes). Responsive → nothing to do. Unresponsive → force-unmount and remount.
2. **Escalate unmounts correctly.** A clean `umount` flushes dirty pages and
   refuses a busy mount (data-safe) but **blocks forever** against a dead
   server. `umount -f` tears down instantly but discards pending writes. So:
   probe liveness first, clean unmount if alive, `umount -f` only if dead —
   then `diskutil unmount force`, then `sudo umount -f` as last resorts.
3. **Reap orphaned `sshfs` processes** bound to the mountpoint. Leaving one is
   what makes the *next* mount fail with a baffling "Operation not permitted".
4. **Check auth before mounting**, so failures say what's wrong rather than
   emerging as a cryptic sshfs error. Two modes: `fnal` (kinit locally if no
   ticket, then GSSAPI) and `master` (no local kinit at all; ride the existing
   ControlMaster, and refuse to proceed non-interactively if it's down).

macOS has no `timeout(1)`, so we use `perl -e 'alarm shift; exec @ARGV'` as a
portable one — the alarm survives the `exec`. On Linux just use `timeout`.

### Two "doctor" scripts for the auth failures that look like something else

- **`lpc-krb-doctor.sh`** — fixes stale Kerberos *delegation*. Symptom:
  `/nashome` gives "Key has expired" or the mount says "Operation not
  permitted", while your Mac `klist` looks perfectly valid. That last bit is the
  trap. Cause: the persistent master delegated whatever ticket existed *when it
  first connected* and keeps reusing that delegation; after the ticket lapses
  and you re-`kinit`, the master is still pinned to the dead one, and LPC's
  krb5-secured NFS rejects you. Fix: `ssh -O exit lpc && ssh lpc true` to force
  re-delegation — which drops any mount riding that master, so the script
  remounts what it dropped.
- **`lxplus-doctor.sh`** — master / AFS-token / mount health for CERN. With a
  live master it first tries a promptless `kinit -R && aklog` **on the node**
  (no 2FA needed, works up to that 5-day renew-until). Only if that fails does
  anyone need to re-login interactively, and it refuses to recycle a live master
  when no terminal is attached.

Both take `--check` for read-only diagnosis. Neither ever runs `kinit` for you.

---

## 6. The golden rules (violate these and you will lose work)

These are the rules we actually operate by. The first two are correctness; the
rest are performance.

### Rule 1 — Git commands that write to `.git/` run *on the cluster*, never through the mount.

SSHFS breaks git's lock-and-rename atomic write. In practice this has truncated
`.git/config` to a stub mid-`branch -f`, losing every configured remote; and on
macOS it scatters `._*` AppleDouble files through `.git/objects`, `refs`, and
`logs` that `git fsck` then reports as `badRefName`.

```bash
# Writes .git/ — run server-side:  commit, branch -f, remote add, config, gc,
#                                  and the ref updates done by fetch / push
ssh lpc 'cd <project> && git commit -am "..."'
ssh lpc 'cd <project> && git push origin <branch>'

# Read-only — through the mount is fine:  log, status, diff, show, branch -vv
```

Convenient side effects: GitHub credentials and `user.name`/`user.email` are
already configured on the cluster, so you don't forward an SSH agent or
duplicate your identity locally.

If `.git/config` does get clobbered: rebuild it on the cluster with
`git remote add` / `git config`, then `find .git -name '._*' -delete`, and
confirm with `git remote -v && git fsck && git branch -vv`.

### Rule 2 — Whole-file rewrites belong on the cluster too.

The same atomicity failure hits *ordinary files*, not just git internals. A
large write, or a search-and-replace-everything edit, can **silently truncate
the target to 0 bytes while reporting success**. We have watched a ~330-line
Python file get emptied by one such edit; the next edit failed "string not
found"; `ssh lpc 'wc -l <file>'` showed 0.

The failure is *intermittent* — small surgical single-line edits through the
mount are fine in practice. It's the whole-file case that's dangerous.

```bash
# Mechanical / whole-file changes: do them server-side
ssh lpc "sed -i 's/old/new/g' <file>"
# or author locally to a temp file and push it over
scp /tmp/newfile.py lpc:<project>/newfile.py

# If you did edit through the mount, verify immediately:
ssh lpc 'wc -l <f> && python -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" <f>'

# If truncated, restore server-side (this writes .git/index, so per Rule 1 it
# must run on the cluster), then redo the change server-side:
ssh lpc 'cd <project> && git checkout HEAD -- <file>'
```

**Create new and large files on the cluster from the start.** A brand-new file
truncated on its very first write has no committed version to restore from.
And **commit often** — a clean server-side commit is what turns a truncation
into a non-event.

### Rule 3 — Never execute anything from the mount.

Architecture mismatch (Apple Silicon arm64 vs the cluster's x86_64 Linux), and
the venvs there symlink into `/cvmfs`, which doesn't exist on your laptop. Those
venvs are real and correct *on the cluster*; they are simply unusable from the
laptop side. All execution goes through `ssh`.

### Rule 4 — No bulk `find` / `grep -r` through the mount.

Every directory read is a network round-trip. Push the work to the server:

```bash
ssh lpc 'cd <project> && git grep <pattern>'
```

Narrow reads and narrow greps through the mount are fine — that's the whole
point of having it. It's recursive scans that hurt.

### Rule 5 — Don't pull big files through the mount.

Anything over ~50 MB (logs, ROOT files, datasets): inspect server-side with
`ssh lpc 'head -n 200 <file>'` or `ssh lpc "awk '...' <file>"`.

### Rule 6 — Don't trust file watchers.

SSHFS doesn't propagate inotify events reliably. Any watch-mode tool
(auto-reload, test watchers, live-preview) should be treated as broken over the
mount.

### Rule 7 — Each `ssh` is a fresh shell.

`cmsenv` and `source setup.sh` do **not** persist between invocations. Chain
everything into one call:

```bash
ssh lpc 'source /cvmfs/cms.cern.ch/cmsset_default.sh \
         && cd <proj>/CMSSW_X_Y_Z/src && cmsenv && scram b -j8'
```

### Rule 8 — Long compute goes to batch, never to the interactive node.

`ssh lpc 'cd <dir> && condor_submit job.sub'`, then `condor_q` to watch.

---

## 7. VS Code configuration

Opening a mounted folder as an ordinary *local* workspace works, but VS Code's
default background behaviour is catastrophic over SSHFS. Three separate
subsystems will each try to enumerate the tree, repeatedly, forever. Here's what
each one does and how we stop it.

### The Git extension

VS Code auto-runs `git status -uall` — enumerate **every untracked file** — on a
timer. Over SSHFS that froze in uninterruptible (`U`) state for over an hour and
the calls stacked up until the extension host jammed and new sessions wouldn't
start at all. Setting `status.showUntrackedFiles=no` in the repo does *not* help,
because VS Code passes `-uall` explicitly.

Since all our real git runs server-side anyway (Rule 1), we just turn it off.

### Pylance / the Python extension

Two distinct offenders. Pylance enumerates the workspace tree (~150 s per pass
over SSHFS — and note `python.analysis.indexing=false` and
`diagnosticMode=openFilesOnly` do **not** stop the enumeration). Separately, the
Python Environments extension's `pet` locator globs for venvs across the tree
(13 s on lxplus; **hangs forever if the mount is dead**), and it also stats
`~/<child>/conda-meta` for every child of your home directory — which means it
walks into the mount points themselves.

That last one bit us badly on 2026-08-18: one dead `~/lpc` mount stalled Python
discovery and therefore *every* Claude Code launch in *every* window, local
projects included, for two hours.

### The file watcher

Futile (Rule 6) and expensive. Exclude the mounts outright.

### Our settings

**User-level** (`~/Library/Application Support/Code/User/settings.json` on macOS;
`~/.config/Code/User/settings.json` on Linux):

```jsonc
{
  // Point Python at a LOCAL interpreter — the mounted Linux venv is unusable here
  "python.defaultInterpreterPath": "/Users/<you>/miniforge3/bin/python",
  "python.analysis.indexing": false,
  "python.analysis.diagnosticMode": "openFilesOnly",
  "python.analysis.autoSearchPaths": false,

  "files.watcherExclude": {
    "/Users/<you>/lpc/**": true,
    "/Users/<you>/nashome/**": true,
    "/Users/<you>/lxplus/**": true,
    "**/*venv*/**": true, "**/__pycache__/**": true, "**/.git/objects/**": true
  },
  "search.followSymlinks": false,

  // Don't reopen mount folders on launch before the mount is up
  "window.restoreWindows": "none",

  // Tame git globally (see above)
  "git.autorefresh": false,
  "git.untrackedChanges": "hidden",
  "git.autofetch": false
}
```

**Per-project**, we use a `~/<PROJECT>.code-workspace` file kept on **local
disk** (not on the mount), one per project, opened via *File → Open Workspace
from File…*. Two reasons it must be a workspace file rather than folder
settings: some of these keys are only honoured at workspace level, and a
`.vscode/settings.json` would otherwise have to live on the mount.

```jsonc
{
  // SIDM on the LPC mount (~/lpc/SIDM -> cmslpc-el9:/uscms_data/d3/<user>/SIDM)
  "folders": [{ "path": "/Users/<you>/lpc/SIDM" }],
  "settings": {
    "python-envs.workspaceSearchPaths": [],   // stop the pet locator globbing the mount
    "python.analysis.exclude": ["**"],        // "**" matches root => pyright does zero readdirs;
                                              // open files keep full IntelliSense
    "python.defaultInterpreterPath": "/Users/<you>/miniforge3/bin/python",

    "git.enabled": false,                     // all real git runs via `ssh lpc`
    "git.autorefresh": false,
    "git.autofetch": false,
    "git.untrackedChanges": "hidden",
    "git.decorations.enabled": false,
    "scm.diffDecorations": "none",

    "files.watcherExclude": {
      "**/.git/**": true, "**/sidm_venv/**": true, "**/*.egg-info/**": true
    }
  }
}
```

Everything here is **scoped to that workspace** — your ordinary local repos keep
full VS Code git integration and full Pylance indexing.

> **Sanity check if it ever feels slow again:** look at the interpreter in the
> status bar. If it shows a path under `~/lpc/` or `~/lxplus/`, a mount venv
> crept back in — re-select the local interpreter and the probing stops.

---

## 8. Claude Code in this setup

**Where it runs:** on the laptop, in the VS Code panel
(`"claudeCode.preferredLocation": "panel"`). Its working directory is whatever
folder the workspace opened — i.e. a path under the mount. So from Claude's
point of view it is editing local files that happen to be slow, and it reaches
the cluster the same way you do, with `ssh lpc '...'` through its Bash tool.

**One setting is mandatory:**

```jsonc
"claudeCode.usePythonEnvironment": false
```

The extension used to `await` the Python extension's `resolveEnvironment()` with
no timeout before spawning Claude. Combined with the `pet` locator walking into
a wedged mount (§7), that stalled *every* Claude launch by 12–14 minutes,
including in purely local windows. Turning it off launches immediately; the only
thing lost is injection of the selected venv into Claude's shell, and the local
conda is on `PATH` anyway.

**Session history is local and durable.** Conversations live in
`~/.claude/projects/-Users-<you>-lpc-<project>/` on the laptop — keyed by the
cwd path, so opening the project through the same workspace file each time keeps
one continuous history. Nothing is written to the cluster. We set
`"cleanupPeriodDays": 1825` so history isn't pruned.

**Per-project `CLAUDE.md`.** Every project on a mount gets a `CLAUDE.md` at its
root telling the agent the rules from §6 — that the files are remote, that
execution goes via `ssh`, that git writes are server-side, that whole-file
rewrites can truncate. We keep a template at `~/lpc-tools/CLAUDE.md.template`
to copy in and fill the paths. Without it the agent will cheerfully run `pytest`
locally against an x86_64 venv, or commit through the mount.

**A `SessionStart` hook does pre-flight checks.** Registered in
`~/.claude/settings.json`:

```jsonc
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{ "type": "command",
                  "command": "/Users/<you>/lpc-tools/session-start-checks.sh",
                  "timeout": 25 }]
    }]
  }
}
```

The script exits silently when nothing is wrong. When something is, it surfaces
it to both you and the agent. It checks:

1. **Globally, regardless of cwd:** is any FUSE-T mount *dead* — still listed in
   `mount` but with no `go-nfsv4`/`sshfs` daemon behind it? This is the
   condition from §7 that stalls every window, so it's worth flagging even in
   local projects. The check reads `mount` and `ps` only — it never touches the
   mount, so it can't itself hang.
2. **If cwd is under a mount:** is that mount active, and is it *responsive*
   (probed with a hard-timeout `ls`, so a wedge is reported rather than silently
   passing a `mount | grep`)?
3. **FNAL cwd:** valid Kerberos ticket (via `/usr/bin/klist -s`, per §3), and a
   live `lpc` ControlMaster.
4. **CERN cwd:** live `lxplus` master — and if it *is* alive, a functional AFS
   check, because a master can be up while the node's token has lapsed. It
   reads your own AFS home specifically, since a tokenless session can still
   read world-readable paths and would fool a naive probe.

Note what it deliberately does *not* do: it never runs `kinit`, never
auto-mounts, never touches the CERN 2FA. It diagnoses and tells you the exact
command. That's a deliberate choice — automation that acts on authentication
state without asking is how you end up with a dropped mount mid-write.

---

## 9. Jupyter: notebook on the mount, kernel on the cluster

The notebook file lives on the cluster (via the mount), the kernel runs on the
cluster, and only the rendering happens locally. One source of truth, no
copy-back.

```bash
# 1. On the cluster, inside tmux so it survives your SSH dropping:
ssh lpc
tmux new -s jup
source <your env setup>
jupyter server --no-browser --port=18888

# 2. The `lpc` alias already forwards localhost:18888 (see §4).

# 3. In VS Code: open the .ipynb from ~/lpc/<project>,
#    Select Kernel -> Existing Jupyter Server -> paste the
#    http://localhost:18888/?token=... URL the server printed.
```

If the kernel "disconnects": check the `jupyter server` is still alive in tmux,
and that the master hasn't lapsed (`ssh lpc true` re-establishes it silently at
FNAL).

---

## 10. Daily routine

```bash
# --- Fermilab ---
/usr/bin/klist -s || /usr/bin/kinit -f <user>@FNAL.GOV   # ~26h ticket
~/lpc-tools/mount.sh                                     # idempotent; self-heals a wedge
# then: File -> Open Workspace from File... -> ~/SIDM.code-workspace

# --- CERN ---
ssh -O check lxplus || ssh lxplus true   # run in a TERMINAL: password + 2FA
~/lpc-tools/mount-lxplus.sh
# then open ~/ETROC-Analysis.code-workspace

# --- before sleeping the laptop / at end of day ---
~/lpc-tools/umount.sh
```

**Order matters: mount first, open the workspace second.** That's why
`window.restoreWindows` is `"none"` — VS Code restoring a mount folder before
the mount is up is exactly how you get a hung window.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Transport endpoint is not connected` after laptop sleep | mount died with the network | `~/lpc-tools/umount.sh && ~/lpc-tools/mount.sh` |
| `ls ~/lpc` hangs, but `mount`/`ssh`/Kerberos all look fine | FUSE-T server died; kernel mount is a corpse (hard, `nointr` NFS) | `umount -f` (no sudo) — plain `umount` and `diskutil` will hang here. Our `umount.sh` does this escalation for you |
| Every VS Code window and every Claude launch is slow, even local ones | one dead mount, being stat'd by the Python locator | same as above; then the settings in §7 prevent recurrence |
| `No Kerberos credentials available` | no ticket, or conda's `klist` lying to you | `/usr/bin/kinit -f <user>@FNAL.GOV` — explicit path (§3) |
| `/nashome` says "Key has expired" but Mac `klist` is valid | master pinned to a stale *delegation* | `~/lpc-tools/lpc-krb-doctor.sh` (or `ssh -O exit lpc && ssh lpc true`, then remount) |
| `~/lxplus` reads → "Permission denied", master looks alive | AFS token lapsed on the node | `~/lpc-tools/lxplus-doctor.sh` — tries a promptless `kinit -R && aklog` first |
| `ssh lxplus` asks for 2FA *again* | the master's TCP connection broke (network change, sleep, node drained) | unavoidable; re-login in a terminal. `ControlPersist yes` minimizes it |
| Mount fails with "Operation not permitted" on retry | orphaned `sshfs` process still bound to the mountpoint | `pkill -f "sshfs .*<mountpoint> "` — our `mount.sh` does this automatically |
| SSH handshake hangs, TCP connects | VPN / security filter (Cisco AnyConnect NVM was our culprit) mangling the SSH protocol | quit the VPN client, or test on a phone hotspot |
| `git fsck` reports `badRefName` | macOS `._*` AppleDouble files in `.git/` | `ssh lpc 'cd <proj> && find .git -name "._*" -delete'`; `-o noappledouble` prevents new ones |
| A file you just edited is 0 bytes | SSHFS write truncation (Rule 2) | restore server-side from HEAD, redo the change server-side |

---

## 12. If this is more machinery than you want

Be honest with yourself about the tradeoff. The SSHFS setup exists because we
need laptop-side tooling operating on cluster files. If you don't:

**VS Code Remote-SSH** is strictly better as an *editing* experience — the
language server, the file watcher, the terminal, and git all run on the cluster,
where the files, `/cvmfs`, and the real venvs are native. No mount latency, full
package-aware IntelliSense, and none of §6 or §7 applies. The catch is that
anything you want running locally can't, and on some clusters the VS Code server
component chews through your home-area quota.

You can also mix: Remote-SSH for heavy editing sessions, the mount for when you
need laptop-side tools. They don't conflict.

---

## 13. Checklist for replicating this

- [ ] Install an SSHFS client for your OS (§3). On Windows, install WSL2 first.
- [ ] Install/verify Kerberos; confirm `kinit` resolves to the *system* one, not conda's.
- [ ] Write the `~/.ssh/config` aliases with `ControlMaster` (§4). Test `ssh lpc true` twice — the second should be instant.
- [ ] Copy `mount.sh` / `umount.sh` and the wrappers; set `FNAL_USER` if your cluster username differs from your laptop username.
- [ ] Mount, and confirm `ls ~/lpc` returns quickly.
- [ ] Apply the VS Code user settings (§7), then make one `.code-workspace` per project on local disk.
- [ ] Select a **local** Python interpreter once per workspace; verify the status bar doesn't show a mount path.
- [ ] Drop a `CLAUDE.md` into each project on the mount (§8).
- [ ] Install the `SessionStart` hook.
- [ ] Read §6 twice. Rules 1 and 2 are the ones that cost real work when ignored.
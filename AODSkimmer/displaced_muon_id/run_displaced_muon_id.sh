#!/usr/bin/env bash
# Usage: run_displaced_muon_id.sh MACRO FILELIST OUTPUT [MAX_EVENTS]
# In a CMSSW runtime on LPC or Condor; all compilation/output stays in the job directory.
set -euo pipefail
macro="$1"; filelist="$2"; output="$3"; limit="${4:--1}"
mkdir -p build
date -u
printf 'macro=%s input=%s output=%s max_events=%s\n' "$macro" "$filelist" "$output" "$limit"
root-config --version
root -l -b <<EOF
gSystem->SetBuildDir("build",true);
if (gROOT->LoadMacro("$macro+") < 0) gSystem->Exit(1);
try { displaced_muon_id("$filelist","$output",$limit); } catch (const std::exception& e) { std::cerr << e.what() << std::endl; gSystem->Exit(1); }
.q
EOF
test -s "$output"


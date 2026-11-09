#!/usr/bin/env python3
"""Count all versioned HLT paths in the original 09-28 signal MINIAOD (LPC only)."""
import argparse
import concurrent.futures
import glob
from pathlib import Path
import time
import traceback


FIELDS = ("present", "accepted", "wasrun", "error")


def initialize():
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gSystem.Load("libFWCoreFWLite")
    ROOT.FWLiteEnabler.enable()
    ROOT.gInterpreter.Declare(r'''
        #include "DataFormats/Common/interface/TriggerResults.h"
        #include "FWCore/Common/interface/TriggerNames.h"
        #include <map>
        #include <string>
        #include <vector>
        struct IDMHLTCounter {
            std::map<std::string, std::vector<unsigned long long>> counts;
            std::vector<std::vector<unsigned long long>*> slots;
            edm::ParameterSetID menu;
            bool first = true;
            void add(const edm::TriggerResults& results, const edm::TriggerNames& names) {
                if (first || menu != results.parameterSetID()) {
                    if (names.size() != results.size()) throw std::runtime_error("HLT menu size mismatch");
                    slots.clear();
                    for (unsigned i = 0; i < results.size(); ++i) {
                        auto& c = counts[names.triggerName(i)];
                        if (c.empty()) c.resize(4, 0);
                        slots.push_back(&c);
                    }
                    menu = results.parameterSetID();
                    first = false;
                }
                if (slots.size() != results.size()) throw std::runtime_error("HLT results size changed");
                for (unsigned i = 0; i < results.size(); ++i) {
                    auto& c = *slots[i];
                    ++c[0];
                    c[1] += results.accept(i);
                    c[2] += results.wasrun(i);
                    c[3] += results.error(i);
                }
            }
        };
    ''')


def count_sample(job):
    import ROOT
    from DataFormats.FWLite import Events, Handle
    sample, files = job
    counter = ROOT.IDMHLTCounter()
    handle = Handle("edm::TriggerResults")
    processed = []
    try:
        for filename in files:
            events = Events(filename)
            expected = int(events.size())
            seen = 0
            for event in events:
                event.getByLabel("TriggerResults", "", "HLT", handle)
                if not handle.isValid():
                    raise RuntimeError("Missing TriggerResults::HLT: " + filename)
                result = handle.product()
                counter.add(result, event.object().triggerNames(result))
                seen += 1
            if seen != expected or seen == 0:
                raise RuntimeError("Incomplete/empty input: %s: %d/%d" % (filename, seen, expected))
            processed.append((filename, seen))
            print("FILE OK %s %d %s" % (sample, seen, filename), flush=True)
        counts = {str(item.first): [int(x) for x in item.second] for item in counter.counts}
        if not counts:
            raise RuntimeError("Empty HLT menu: " + sample)
        total = sum(n for _, n in processed)
        for name, (present, accepted, wasrun, error) in counts.items():
            if not (0 <= accepted <= wasrun <= present <= total and 0 <= error <= wasrun):
                raise RuntimeError("Invalid counters for " + name)
        return sample, processed, counts, None
    except Exception:
        return sample, processed, {}, traceback.format_exc()


def save_root(path, results, overall):
    import ROOT
    output = ROOT.TFile.Open(str(path), "RECREATE")
    if not output or output.IsZombie():
        raise RuntimeError("Cannot create ROOT output")
    def write_counts(directory, counts):
        directory.cd()
        names = sorted(counts)
        for column, field in enumerate(FIELDS):
            hist = ROOT.TH1D(field, "Raw unweighted HLT event counts;Versioned HLT path;Events", len(names), 0, len(names))
            for i, name in enumerate(names, 1):
                hist.GetXaxis().SetBinLabel(i, name)
                hist.SetBinContent(i, counts[name][column])
                hist.SetBinError(i, counts[name][column] ** 0.5)
            hist.Write()
    write_counts(output.mkdir("overall"), overall)
    samples = output.mkdir("samples")
    output.cd()
    coverage = ROOT.TH1D("events_per_sample", "Input coverage;Sample;Events", len(results), 0, len(results))
    file_coverage = ROOT.TH1D("files_per_sample", "Input coverage;Sample;Files", len(results), 0, len(results))
    for i, (sample, processed, counts, _) in enumerate(sorted(results), 1):
        directory = samples.mkdir(sample)
        write_counts(directory, counts)
        directory.cd()
        ROOT.TNamed("inputs", "\n".join("%s\t%d" % (p, n) for p, n in processed)).Write()
        coverage.GetXaxis().SetBinLabel(i, sample)
        coverage.SetBinContent(i, sum(n for _, n in processed))
        file_coverage.GetXaxis().SetBinLabel(i, sample)
        file_coverage.SetBinContent(i, len(processed))
    output.cd()
    coverage.Write()
    file_coverage.Write()
    ROOT.TNamed("definition", "TriggerResults::HLT; original 2026-09-28 signal MINIAOD; no cuts; raw unweighted event counts; full versioned paths kept separate; absent paths excluded from present; overlapping triggers counted independently").Write()
    output.Close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--test", action="store_true", help="Timing/runtime test: only the first original file")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    output = run / "trigger_counts.root"
    table = run / "trigger_counts.md"
    if output.exists() or table.exists():
        raise RuntimeError("Refusing to overwrite outputs")
    repo = Path(__file__).resolve().parents[2]
    lists = sorted(glob.glob(str(repo / "AODSkimmer/fileLists/signal/2022EE/iDMmu/*_flist.txt")))
    jobs = []
    all_files = []
    for source in lists:
        files = [line.strip() for line in Path(source).read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        if len(files) != 4 or any("260928_" not in p or not p.endswith(".root") for p in files):
            raise RuntimeError("Not the expected four-file 09-28 signal list: " + source)
        jobs.append((Path(source).name.removesuffix("_flist.txt"), files))
        all_files.extend(files)
    if len(jobs) != 72 or len(all_files) != 288 or len(set(all_files)) != 288:
        raise RuntimeError("Expected 72 samples and 288 unique original inputs")
    if args.test:
        jobs = [(jobs[0][0], jobs[0][1][:1])]
    start = time.monotonic()
    results = []
    # ROOT/FWLite are loaded only in each process, after process creation.
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        for result in pool.map(count_sample, jobs):
            results.append(result)
            sample, processed, counts, error = result
            print("SAMPLE %s files=%d events=%d paths=%d status=%s" %
                  (sample, len(processed), sum(n for _, n in processed), len(counts), "FAIL" if error else "OK"), flush=True)
            if error:
                print(error, flush=True)
    failed = [r[0] for r in results if r[3]]
    total = sum(n for _, processed, _, _ in results for _, n in processed)
    if failed or (not args.test and (total != 720000 or any(sum(n for _, n in r[1]) != 10000 for r in results))):
        raise RuntimeError("Incomplete scan: failed=%r, processed events=%d; no final output published" % (failed, total))
    overall = {}
    for _, _, counts, _ in results:
        for name, values in counts.items():
            dest = overall.setdefault(name, [0] * 4)
            for i, value in enumerate(values):
                dest[i] += value
    initialize()
    save_root(output, results, overall)
    with table.open("x") as target:
        target.write("# HLT counts%s\n\n" % (" (timing test only; not study evidence)" if args.test else ""))
        target.write("Original 2026-09-28 signal MINIAOD; %d events, %d files, %d samples. Raw, unweighted, no cuts.\n\n" %
                     (total, sum(len(r[1]) for r in results), len(results)))
        target.write("Each full versioned path is separate. Overlapping triggers count independently. Present means in that event's HLT menu; absent is not a rejection.\n\n")
        target.write("| HLT path | Accepted events | Present events | Was run | Error |\n|---|---:|---:|---:|---:|\n")
        for name in sorted(overall, key=lambda n: (-overall[n][1], n)):
            present, accepted, wasrun, error = overall[name]
            target.write("| `%s` | %d | %d | %d | %d |\n" % (name, accepted, present, wasrun, error))
    print("COMPLETE samples=%d files=%d events=%d paths=%d elapsed_seconds=%.2f root=%s table=%s" %
          (len(results), sum(len(r[1]) for r in results), total, len(overall), time.monotonic() - start, output, table), flush=True)


if __name__ == "__main__":
    main()

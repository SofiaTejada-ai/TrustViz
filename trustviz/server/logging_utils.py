import csv, os, time
#so you can report results
def log_roc_run(path, n, truth_auc, llm_auc, decision):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = [time.strftime("%Y-%m-%d %H:%M:%S"), n, truth_auc, llm_auc if llm_auc is not None else "", decision]
    header = ["timestamp","n","truth_auc","llm_auc","decision"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f);
        if write_header: w.writerow(header)
        w.writerow(row)

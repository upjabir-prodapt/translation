import json
import os

def generate_report():
    issues_path = '/home/jupyter/Translation/sonar_issues.json'
    measures_path = '/home/jupyter/Translation/sonar_measures.json'
    report_path = '/home/jupyter/Translation/sonarqube_detailed_report.md'

    if not os.path.exists(issues_path) or not os.path.exists(measures_path):
        print("Missing data files.")
        return

    with open(issues_path, 'r') as f:
        issues_data = json.load(f)
    
    with open(measures_path, 'r') as f:
        measures_data = json.load(f)

    report = []
    report.append("# SonarQube Detailed Report: Translation Project")
    report.append(f"**Date:** 2026-05-08")
    report.append("")

    # Summary Metrics (from measures_data base component or issues_data total)
    total_issues = issues_data.get('total', 0)
    report.append("## Summary Metrics")
    report.append(f"- **Total Issues:** {total_issues}")
    
    # Issues breakdown
    types = {}
    severities = {}
    for issue in issues_data.get('issues', []):
        t = issue.get('type')
        s = issue.get('severity')
        types[t] = types.get(t, 0) + 1
        severities[s] = severities.get(s, 0) + 1
    
    report.append("### Issues by Type")
    for t, count in types.items():
        report.append(f"- **{t}:** {count}")
    
    report.append("### Issues by Severity")
    for s, count in severities.items():
        report.append(f"- **{s}:** {count}")
    
    report.append("")
    report.append("## Top Issues (First 20)")
    report.append("| Severity | Type | Component | Message |")
    report.append("|---|---|---|---|")
    for issue in issues_data.get('issues', [])[:20]:
        sev = issue.get('severity')
        typ = issue.get('type')
        comp = issue.get('component', '').split(':')[-1]
        msg = issue.get('message')
        report.append(f"| {sev} | {typ} | {comp} | {msg} |")

    report.append("")
    report.append("## File Coverage Breakdown (Top 20 Files by Size)")
    report.append("| File | Lines of Code | Coverage | Complexity |")
    report.append("|---|---|---|---|")
    
    components = measures_data.get('components', [])
    # Filter for files and sort by ncloc
    files = [c for c in components if c.get('qualifier') == 'FIL']
    
    def get_metric(comp, key):
        for m in comp.get('measures', []):
            if m.get('metric') == key:
                return m.get('value')
        return "N/A"

    files.sort(key=lambda x: int(get_metric(x, 'ncloc') if get_metric(x, 'ncloc') != 'N/A' else 0), reverse=True)

    for f in files[:20]:
        name = f.get('path', f.get('name'))
        ncloc = get_metric(f, 'ncloc')
        cov = get_metric(f, 'coverage')
        compl = get_metric(f, 'complexity')
        report.append(f"| {name} | {ncloc} | {cov}% | {compl} |")

    with open(report_path, 'w') as f:
        f.write("\n".join(report))
    
    print(f"Report generated at: {report_path}")

if __name__ == "__main__":
    generate_report()

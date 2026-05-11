import json
import os
from collections import defaultdict

def generate_full_report():
    issues_path = '/home/jupyter/Translation/sonar_issues.json'
    all_report_path = '/home/jupyter/Translation/ALL_SONAR_ISSUES.md'
    doc_report_path = '/home/jupyter/Translation/DOCTRANSLATOR_ISSUES.md'

    if not os.path.exists(issues_path):
        print("Missing data files.")
        return

    with open(issues_path, 'r') as f:
        issues_data = json.load(f)

    issues = issues_data.get('issues', [])
    
    files_dict = defaultdict(list)
    doc_files_dict = defaultdict(list)
    
    for issue in issues:
        comp = issue.get('component', '').split(':')[-1]
        if 'src/doctranslator/' in comp:
            doc_files_dict[comp].append(issue)
        else:
            files_dict[comp].append(issue)

    def write_report(data_dict, path, title, total):
        report = [f"# {title}", f"**Total Issues:** {total}", "", "---"]
        sorted_files = sorted(data_dict.keys(), key=lambda x: len(data_dict[x]), reverse=True)
        for file_path in sorted_files:
            file_issues = data_dict[file_path]
            report.append(f"## {file_path} ({len(file_issues)} issues)")
            report.append("| Severity | Line | Message | Type |")
            report.append("|---|---|---|---|")
            file_issues.sort(key=lambda x: x.get('line', 0))
            for issue in file_issues:
                report.append(f"| {issue.get('severity')} | {issue.get('line', 'N/A')} | {issue.get('message')} | {issue.get('type')} |")
            report.append("")
        with open(path, 'w') as f:
            f.write("\n".join(report))

    write_report(files_dict, all_report_path, "SonarQube Issues (Filtered - No doctranslator)", len([i for i in issues if 'src/doctranslator/' not in i.get('component', '')]))
    write_report(doc_files_dict, doc_report_path, "SonarQube Issues (doctranslator only)", len([i for i in issues if 'src/doctranslator/' in i.get('component', '')]))
    
    print("Reports generated.")

if __name__ == "__main__":
    generate_full_report()

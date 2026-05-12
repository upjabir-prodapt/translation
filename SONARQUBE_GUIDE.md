# SonarQube Analysis Guide

This guide documents the process for running SonarQube analysis, generating coverage reports, and creating human-readable summaries for the **DocTranslator** project.

## 1. Prerequisites
- **SonarQube Server:** Must be running (default: `http://localhost:9000`).
- **Sonar Scanner:** Path to the executable (e.g., `/home/jupyter/sonar-scanner-5.0.1.3006-linux/bin/sonar-scanner`).
- **Dependencies:** Ensure `uv` or `pip` is installed to run tests.

## 2. Token Generation
If you need a new authentication token, use the following `curl` command (using default admin credentials):

```bash
# Revoke existing (optional)
curl -u admin:admin -X POST "http://localhost:9000/api/user_tokens/revoke?name=mytoken"

# Generate new token
curl -u admin:admin -X POST "http://localhost:9000/api/user_tokens/generate?name=mytoken"
```
Update `sonar.login` in `sonar-project.properties` with the returned token.

## 3. Running Coverage
Tests must be run first to generate the `coverage.xml` file required by SonarQube.

```bash
cd Translation
# Create .env from example if not present
cp .env.example .env

# Run tests with coverage
uv run pytest --cov=src --cov-report=xml:coverage.xml
```

## 4. Running SonarQube Scan
Execute the scanner from the project root. Ensure `sonar-project.properties` is correctly configured with `sonar.python.version=3.12` for modern syntax support.

```bash
cd Translation
/home/jupyter/sonar-scanner-5.0.1.3006-linux/bin/sonar-scanner
```

## 5. Generating Local Reports
The project includes scripts to fetch data from the SonarQube API and generate Markdown reports.

### Step A: Fetch Data from API
```bash
# Fetch Issues
curl -u admin:admin "http://localhost:9000/api/issues/search?projectKeys=Translation&ps=500" -o Translation/sonar_issues.json

# Fetch Measures (Coverage, Bugs, etc.)
curl -u admin:admin "http://localhost:9000/api/measures/component?component=Translation&metricKeys=bugs,vulnerabilities,code_smells,coverage,duplicated_lines_density" -o Translation/sonar_measures.json
```

### Step B: Run Python Scripts
```bash
cd Translation
python3 generate_sonar_report.py
python3 generate_full_report.py
```

## 6. Output Files
- **`sonarqube_detailed_report.md`**: Executive summary of metrics and top issues.
- **`ALL_SONAR_ISSUES.md`**: Detailed list of all issues (filtered).
- **`DOCTRANSLATOR_ISSUES.md`**: Issues specific to the `doctranslator` core package.
- **SonarQube Dashboard**: [http://localhost:9000/dashboard?id=Translation](http://localhost:9000/dashboard?id=Translation)

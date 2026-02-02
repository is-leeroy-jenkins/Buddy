
###### Buddy
![](https://github.com/is-leeroy-jenkins/Buddy/blob/main/resources/Buddy_Project.png)
Budget Buddy is a Streamlit-based analytical assistant designed to support U.S. Federal budget
analysts, financial managers, and policy professionals. It integrates large language models (LLMs),
retrieval-augmented generation (RAG), semantic search, and structured prompt governance to assist
with interpretation of budget guidance, execution data, and authoritative financial policy
documents.

The application is integrated with **GPT-5.2**, including **fine-tuned LLMs hosted on Hugging Face
**, trained on curated budgetary, financial, and policy datasets—particularly those reflected in the
guidance tables and execution data described below.

---

## Core Capabilities

* Conversational analysis of federal budget and financial guidance
* Source-aware responses with traceable references
* Retrieval-augmented generation using user-provided documents
* Semantic search across guidance and execution datasets
* Prompt and system-instruction version control
* Exportable analytical outputs (Markdown, PDF)

---

## Model Integration

### GPT-5.2 (Primary Inference Engine)

Budget Buddy uses **GPT-5.2** via the OpenAI Responses API as its primary inference engine. The
application is structured to support:

* Structured response outputs (text, sources, analytical artifacts)
* Tool-generated analysis (tables, derived files)
* Prompt versioning via OpenAI Prompt IDs

### Fine-Tuned Models on Hugging Face

In addition to base GPT-5.2 capabilities, Budget Buddy is designed to leverage **fine-tuned large
language models hosted on Hugging Face**, trained on:

* Federal budget execution data
* Appropriations and fiscal law guidance
* DoD-specific and government-wide policy documentation
* Structured tabular datasets used in budget reporting and analysis

These fine-tuned models improve:

* Domain-specific accuracy
* Terminology alignment (OMB, DoD, Treasury, GAO)
* Consistency when answering budget and execution questions

---

## Guidance Corpus (Current)

Budget Buddy is explicitly designed to operate against authoritative guidance, including—but not
limited to—the following categories reflected in the attached tables:

## Federal Budget Guidance

- Financial guidance common across all federal agencies.

| File Name | Description |
|----------|-------------|
| Balanced Budget and Emergency Deficit Control Act of 1985 | Establishes statutory limits on federal spending and deficit control mechanisms, including sequestration procedures. |
| Budget Control Act of 2011 | Sets discretionary spending caps and establishes enforcement mechanisms to control federal deficits. |
| Digital Accountability And Transparency Act of 2014 | Requires standardized federal spending data and improved transparency through government‑wide financial reporting. |
| DOD Logo Overview | Provides official guidance on the authorized use and representation of Department of Defense logos and insignia. |
| Federal Account Symbols And Titles Book | Defines Treasury account symbols and official titles used for federal budgetary and accounting purposes. |
| Federal Acquisition Regulation | Establishes uniform policies and procedures governing the acquisition of goods and services by federal agencies. |
| Federal Government Standards For Internal Controls | Defines the internal control framework for federal agencies to ensure accountability, integrity, and compliance. |
| Federal Managers Financial Integrity Act of 1982 | Requires agencies to establish internal controls and report annually on their effectiveness. |
| Federal Trust Fund Accounting Guide | Provides accounting guidance for the management and reporting of federal trust funds. |
| Financial Management Regulations DOD 7000‑14‑R | Establishes DoD‑specific financial management policies, procedures, and accounting requirements. |
| Fiscal Responsibility Act | Establishes statutory measures intended to improve fiscal discipline and control federal spending. |
| Government Auditing Standards | Sets professional standards for audits of government organizations, programs, activities, and functions. |
| Government Invoicing User Guide | Provides guidance on federal invoicing standards and processes for government transactions. |
| Government Performance and Results Act of 1993 | Requires agencies to engage in strategic planning and performance measurement to improve program effectiveness. |
| GPRA Modernization Act of 2010 | Updates GPRA by strengthening performance management, cross‑agency goals, and accountability. |
| Incident Management Handbook | Provides guidance for managing and coordinating responses to incidents affecting government operations. |
| Independent Offices Appropriation Act of 1953 | Establishes authorities related to user fees and charges for services provided by federal agencies. |
| Inspector General Act of 1978 | Establishes Offices of Inspector General to conduct audits and investigations promoting efficiency and integrity. |
| OMB Circular A‑11 Preparation Submission And Execution Of The Budget | Provides comprehensive guidance for preparing, submitting, and executing the President’s Budget. |
| OMB Circular A‑11 Section 120 Apportionment Process | Defines the apportionment process used to control the rate of obligation of budgetary resources. |
| OMB Circular A‑19 Legislative Coordination and Clearance | Establishes procedures for coordinating and clearing agency legislative proposals and testimony. |
| OMB Circular A‑25 User Charges | Establishes federal policy for assessing user charges for government goods and services. |
| OMB Circular A‑45 Rental and Construction of Government Quarters | Provides guidance on rental rates and construction policies for government‑owned housing. |
| OMB Circular A‑76 Performance of Commercial Activities | Establishes policy for determining whether commercial activities should be performed by the government or private sector. |
| OMB Circular A‑94 Discount Rates for Time‑Distributed Costs and Benefits | Provides discount rates and analytical guidance for benefit‑cost analysis of federal programs. |
| OMB Circular A‑97 Services for State and Local Governments | Establishes policies for providing federal services to state and local governments. |
| OMB Circular A‑123 Managements Responsibility for Enterprise Risk Management and Internal Control | Defines management responsibilities for internal control and enterprise risk management across federal agencies. |
| OMB Circular A‑136 Financial Reporting Requirements | Establishes requirements for federal agency financial statements and reporting. |
| Principles Of Federal Appropriations Law Volume One | Authoritative GAO guidance on foundational principles governing the use of federal appropriations. |
| Principles Of Federal Appropriations Law Volume Two | Expands on appropriations law with detailed case law, interpretations, and applications. |
| Statements of Federal Federal Financial Accounting Concepts and Standards | Establishes accounting concepts and standards for federal financial reporting. |
| The Anti‑Deficiency Act PL 97‑258 | Prohibits federal agencies from obligating or expending funds in excess of appropriations or before enactment. |
| The Anti‑Deficiency Reform and Enforcement Act of 2018 | Strengthens Anti‑Deficiency Act enforcement and reporting requirements to improve fiscal accountability. |
| The Budget And Accounting Act of 1921 | Establishes the federal budget process and created the Bureau of the Budget and the General Accounting Office. |
| The Chief Financial Officers Act of 1990 | Establishes agency Chief Financial Officers and modernizes federal financial management practices. |
| The Congressional Budget and Impoundment Control Act of 1974 | Establishes the congressional budget process and restricts executive impoundment of appropriated funds. |
| The Economy Act of 1932 | Authorizes interagency agreements for the provision of goods and services on a reimbursable basis. |
| The Stafford Act | Provides the statutory framework for federal disaster response and emergency assistance. |
| The Supplemental Appropriations Act 1955 | Provides additional appropriations authority beyond regular annual funding acts. |
| Title 2 Code of Federal Regulations – Uniform Administrative Requirements, Cost Principles, and Audit | Establishes uniform administrative, cost, and audit requirements for federal financial assistance. |
| Title 31 Code of Federal Regulations – Money and Finance | Codifies Treasury and federal financial management regulations governing money and finance. |
| US Standard General Ledger Account Definitions | Defines standardized account structures used for federal accounting and financial reporting. |

## Department of War Guidance

- Support for DoD-specific budget formulation, execution, audit, and compliance analysis.

| File Name | Description |
|----------|-------------|
| Access Management For DOD Information Systems DOD 8520‑04  | Establishes Department of Defense policy for identity, credential, and access management (ICAM) across DoD information systems. |
| Acquisition Management Of Contractor‑Prepared Data DOD 5010‑12  | Defines requirements for managing, retaining, and accessing data produced by contractors during DoD acquisitions. |
| Acquisition Transformation Strategy  | Outlines DoD‑wide strategic objectives for modernizing acquisition processes, governance, and workforce practices. |
| Administrative Instruction DODAM 28  | Provides administrative procedures and internal management guidance under the DoD Administrative Management framework. |
| Audit Manual DOD 7600‑07  | Establishes audit policies, responsibilities, and procedures for DoD financial and performance audits. |
| Civilian Personnel Management System Payroll DOD 1400‑25‑v550  | Governs payroll administration within the DoD Civilian Personnel Management System (CPMS). |
| Civilian Personnel Management System Performance DOD 1400‑25‑v430  | Defines performance management policies for DoD civilian employees, including evaluation and appraisal standards. |
| Cybersecurity Reciprocity Playbook DOD 2024‑01‑02  | Provides guidance for reciprocal acceptance of cybersecurity authorizations across DoD components to reduce duplication. |
| DevSecOps Fundamentals v2.5  | Introduces core principles, practices, and terminology for implementing DevSecOps in DoD environments. |
| DevSecOps OSS Development  | Addresses the development and use of open‑source software within DoD DevSecOps pipelines. |
| DevSecOps Playbook DODCIO 2021‑10‑19  | Official DoD CIO guidance for planning, implementing, and scaling DevSecOps capabilities. |
| DevSecOps Reference Design AWS Managed Services DODCIO 2021‑10‑19  | Provides a reference architecture for deploying DevSecOps using AWS managed services in DoD contexts. |
| DNS IP Address Use And Approval DOD 8410  | Establishes policy for allocation, registration, and approval of DNS names and IP address usage within DoD networks. |
| DOD Cloud Reference Design Cloud GitHub Azure  | Reference architecture describing DoD‑approved cloud implementation patterns using Microsoft Azure. |
| DOD AI Strategy  | Defines DoD objectives, priorities, and governance principles for artificial intelligence adoption. |
| Emergency Management Program DOD 6055‑17  | Establishes policy for preparedness, response, recovery, and mitigation of emergencies affecting DoD operations. |
| Forms Management Program DOD 7750‑08  | Governs the lifecycle management of DoD forms, including standardization and control. |
| Information Network Transport DOD 8010‑01  | Provides policy for managing DoD network transport infrastructure and communications pathways. |
| Information Technology Standards DOD 8310‑01  | Establishes requirements for adopting and enforcing IT standards across the Department of Defense. |
| Infosec Program and Securing SCI DOD 5200‑01p  | Governs information security programs, including the protection of Sensitive Compartmented Information (SCI). |
| Management Of DOD IE DOD 8000‑01  | Establishes governance for managing the DoD Information Enterprise (IE). |
| Multifactor Authentication DOD Networks  | Defines requirements for implementing MFA across DoD network environments to strengthen access security. |
| National Defense Strategy 2026  | Sets strategic defense priorities, threats, and force‑planning guidance for the Department of Defense. |
| Online Information Management And Electronic Messaging DOD 8170‑01  | Establishes policy for managing official DoD online content and electronic communications. |
| OPSEC Manual DOD 5205‑02  | Provides policy and procedures for Operations Security to protect critical information from adversaries. |
| Personnel Identity Protection Program DOD 1000‑25  | Establishes policies for identity protection and credentialing of DoD personnel. |
| PPBE Reform Activities FY2026  | Describes planned reforms to the Planning, Programming, Budgeting, and Execution (PPBE) process for FY2026. |
| Records Management Standards DTM 22‑001  | Establishes mandatory standards for managing, retaining, and disposing of DoD records. |
| Use Of Non‑Government Owned Mobile Devices.  | Provides policy governing the use of personally owned mobile devices for official DoD activities. |


## File A (Account Balances)

- File A (Account Balances) is part of the package of data submitted to USAspending.gov every month by
federal agencies, as required by the DATA Act. As part of the monthly submission process, agencies
generate File A automatically from data in the Governmentwide Treasury Account Symbol Adjusted Trial
Balance System (GTAS), or choose to upload their own custom File A data. 

- File A contains budgetary resources, obligation, and outlay data for all the relevant Treasury Account Symbols (TAS) in a
reporting agency, with additional breakdown by Budget Function. It includes both award and non-award
spending (grouped together), and crosswalks with the SF 133 report.

## SF 133 Report on Budget Execution and Budgetary Resources

- The SF 133 Report on Budget Execution and Budgetary Resources fulfills the requirement in 31 U.S.C.
1511 - 1514 that the President review Federal expenditures at least four times a year. 

- SF 133s provide historical reference that can be used to help prepare the President's Budget, program
operating plans, and spend-out rate estimates. 

- Agencies submit the data that appear on these reports to the Department of the Treasury Bureau of Fiscal Service. While OMB publishes these reports as a
service to agency budget and finance offices and other interested parties, the underlying data is
submitted by the agencies.


---

## Application Features

### 1. Conversational Chat Interface

* Analyst-oriented natural language interaction
* Execution modes:

    * **Standard**
    * **Guidance Only**
    * **Analysis Only**
* Custom chat avatars and branding support
* Session-scoped conversational context

### 2. Source Attribution & Guidance Review

* Automatic extraction of source references
* Display of document titles, snippets, and links
* Supports auditability and defensible analysis

### 3. Analytical Artifacts

* Tables and text generated by model-driven analysis
* Downloadable files produced during reasoning
* Dedicated Analysis tab for structured outputs

### 4. Prompt & System Instruction Management

* SQLite-backed prompt repository
* Create, edit, load, and version system instructions
* Convert between:

    * XML-delimited instruction blocks
    * Markdown representations

This enables controlled experimentation and governance of AI behavior.

### 5. Retrieval-Augmented Generation (RAG)

* Upload reference documents
* Chunk and inject relevant context into prompts
* Ground responses in authoritative material rather than model priors alone

### 6. Semantic Search

* Sentence-level embeddings
* SQLite-backed vector storage
* Cosine similarity scoring
* Reusable embedded corpora across sessions

### 7. Export & Reporting

* Export system instructions as XML or Markdown
* Export chat history as Markdown or PDF
* Designed for briefings, documentation, and archival use

---

## Data Storage & Architecture

* Local SQLite database for:

    * Prompt storage
    * Semantic embeddings
    * (Planned) chat history persistence
* Modular design anticipates:

    * External vector databases
    * Centralized prompt registries
    * Multi-user or shared analytical environments

---

## Intended Users

* Federal budget analysts
* Financial management professionals
* DoD and civilian agency policy analysts
* Data scientists supporting budget formulation and execution
* Program and portfolio analysts requiring explainable AI assistance

---

## Disclaimer

Budget Buddy is an analytical support tool. Outputs generated by large language models should be
independently reviewed and validated by qualified personnel before use in official decisions,
submissions, or policy actions.

## 📝 License

- Buddy is published under
  the [MIT General Public License v3](https://github.com/is-leeroy-jenkins/Boo/blob/main/LICENSE).

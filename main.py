from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# with open('ipc_law.txt', 'r') as file:
#     ipc_law = file.read()

tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
model = AutoModelForTokenClassification.from_pretrained("law-ai/InLegalBERT")

data = """AMUHACKS 3.0 Rules and Guidelines Eligibility:
1. Par�cipants: AMUHACKS 3.0 welcomes par�cipants from all academic backgrounds,
genders, and loca�ons. There are no age restric�ons. Should be any university course at an
accredited ins�tu�on na�onwide.
2. Team Forma�on: Teams can comprise 2 to 4 members. Excep�ons for larger teams may be
considered on a case-by-case basis upon contac�ng the organizing commitee.
3. University Enrollment: Par�cipants must be currently enrolled in any university course to be
eligible to par�cipate.
Project Development:
1. Development Start: Development may only commence at the official start �me of the event.
Any prior development will result in automa�c disqualifica�on.
2. Tools and Languages: Par�cipants are free to use any so�ware development tools,
programming languages, or frameworks for their projects.
3. GitHub Repository: Teams must create a GitHub repository for their project a�er April 25th,
2024. The repository name should be in the format "Teamname_amuhacks3.0" and it should
be public. In the README file of the repository, include the following hashtags:
#AMUHACKS3.0 #GDSCAMU #CSSAMU #CSDAMU #AMU.
4. Project Submission: Teams must submit the link to their GitHub repository via the circulated
form provided by the organizing commitee. Regular commits to the team repository are
required throughout the event. And the project must be hosted.
5. Code Review: All submited projects will undergo random code reviews to ensure originality
and compliance with the rules.
6. One Entry per Team: Each team may submit only one project for considera�on. Par�cipa�on
is on a per-team basis; par�cipants cannot be part of more than one team.
Atendee Code of Conduct:
1. Harassment-Free Environment: AMUHACKS 3.0 is dedicated to providing a harassment-free
experience for all par�cipants. Harassment of any form, including but not limited to gender,
gender iden�ty, age, sexual orienta�on, disability, race, ethnicity, na�onality, or religion, will
not be tolerated.
2. Respec�ul Behavior: Par�cipants are expected to treat each other with respect and refrain
from engaging in any behavior that may cause discomfort or offense to others.
3. Consequences: Viola�on of the code of conduct may result in expulsion from the hackathon
without refund, and further ac�on may be taken as deemed appropriate by the organizers.
By par�cipa�ng in AMUHACKS 3.0, all atendees agree to abide by these rules and guidelines. Let's
make this event a collabora�ve and enriching experience for everyone involved!
Date: April 27th-28th, 2024 (24-Hour Online Event)
Organized by: Department of Computer Science in associa�on with GDSC AMU and CSS."""


nlp = pipeline("document-question-answering", model=model, tokenizer=tokenizer)
example = "Summerize important parts?"

ner_results = nlp(data, example)
print(ner_results)

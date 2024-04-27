from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(reference, candidate):
    # Split the sentences into words
    reference = reference.split()
    candidate = candidate.split()

    # Calculate the BLEU score
    score = sentence_bleu([reference], candidate, weights=(1, 0, 0, 0))

    return score

# Test the function
reference = "elon university is on the selective side, with an acceptance rate of 45% as of 2014. the university includes elon college, the college of arts and sciences; the martha and spencer love school of business; the school of communications; the school of education; the school of law; and the school of health sciences."
candidate = "Elon University offers a variety of academic programs through several schools, including Elon College, the College of Arts and Sciences; the Martha and Spencer Love School of Business; the School of Communications; the School of Education; the School of Law; and the School of Health Sciences. The university had an acceptance rate of 45% in 2014, indicating a moderately selective admission process"
candidate2 = "Elon University, with an acceptance rate of 45% as of 2014, includes Elon College, the College of Arts and Sciences; the Martha and Spencer Love School of Business; the School of Communications; the School of Education; the School of Law; and the School of Health Sciences."
print(f"Before prompting: {calculate_bleu(reference, candidate)}")
print(f"After prompting: {calculate_bleu(reference, candidate2)}")
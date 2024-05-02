from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(reference, candidate):
    # Split the sentences into words
    reference = reference.split()
    candidate = candidate.split()

    # Calculate the BLEU score
    score = sentence_bleu([reference], candidate, weights=(1, 0, 0, 0))

    return score

# Test the function
reference = "llanthony priory is a picturesque, partly ruined former augustinian priory in the beautiful and secluded vale of ewyas, a steep sided once glaciated valley within the black mountains area of the brecon beacons national park in monmouthshire, south east wales."
candidate = "llanthony priory is a partly ruined former augustinian priory in the vale of ewyas, a steep sided once glaciated valley within the black mountains area of the brecon beacons national park in monmouthshire, south east wales."
candidate2 = "Elon University, with an acceptance rate of 45% as of 2014, includes Elon College, the College of Arts and Sciences; the Martha and Spencer Love School of Business; the School of Communications; the School of Education; the School of Law; and the School of Health Sciences."
print(f"Before prompting: {calculate_bleu(reference, candidate)}")
print(f"After prompting: {calculate_bleu(reference, candidate2)}")
import re

OPTIONS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]

# ResponseFormatPrompts = {
#     'none': '',
#     'mcq_cot': "First provide the intermediate reasoning process that leads to the correct answer, then give the correct option letter in the format 'ANSWER: X'.",
#     'mcq_direct': "Please answer using the option's letter from the given choices directly.",
#     'mcq_direct2': "Directly answer with the letter.",
#     'vqa_cot': "Please provide the intermediate reasoning process that leads to the correct answer.",
#     'vqa_cot2': "First provide the intermediate reasoning process that leads to the correct answer, then give the final answer in the format 'ANSWER: XXX'.",
#     'vqa_direct': "Please answer the question using a single word or phrase.",
#     'vqa_counting': "Please answer the question using an integer number.",
#     'vqa_yn': "Answer yes or no.",
# }

# 1017
# ResponseFormatPrompts = {
#     'none': '',
#     'mcq_cot2': "Please provide the intermediate reasoning process and then give the correct option letter in the format 'ANSWER: X'.",  # for more complex question
#     'mcq_cot': "Please provide the rationale for your answer and then give the correct option letter in the format 'ANSWER: X'.",
#     'mcq_direct': "Please answer the question directly using the option's letter from the given choices.",
#     'vqa_cot2': "Please provide the intermediate reasoning process and then give the final answer in the format 'ANSWER: XXX'.",
#     'vqa_cot': "Please provide the rationale for your answer and then give the final answer in the format 'ANSWER: XXX'.",
#     'vqa_direct': "Please answer the question directly using a single word or phrase.",
#     'vqa_counting': "Please answer the question directly using a single number.",
#     'vqa_yn': "Please answer the question directly using yes or no.",
# }

# # 1020
# ResponseFormatPrompts = {
#     # 'mcq_cot2': "Please first provide the rationale for your answer, and then give the correct option letter in the format 'ANSWER: X'.",
#     'mcq_cot': "Please first provide the intermediate reasoning process that leads to the correct answer, and then give the correct option letter in the format 'ANSWER: X'.",
#     'vqa_cot2': "Please first provide the rationale for your answer, and then give the final answer in the format 'ANSWER: XXX'.",
#     'vqa_cot': "Please first provide the intermediate reasoning process that leads to the correct answer, and then give the final answer in the format 'ANSWER: XXX'.",
#     'mcq_direct': "Please answer the question using the option's letter from the given choices.",
#     'vqa_direct': "Please answer the question using a single word or phrase.",
#     'vqa_counting': "Please answer the question using a single number.",
#     'vqa_yn': "Please answer the question using yes or no.",
#     'none': '',
# }

# 1028
ResponseFormatPrompts = {
    # 'mcq_cot2': "Please first provide the rationale for your answer, and then give the correct option letter in the format 'ANSWER: X'.",
    # 'vqa_cot2': "Please first provide the rationale for your answer, and then give the final answer in the format 'ANSWER: XXX'.",
    # Please perform detailed reasoning first and then provide the correct answer in the format 'ANSWER: X'.
    'mcq_cot': "Please first provide the reasoning process that leads to the correct answer, and then give the correct option letter in the format 'ANSWER: X'.",
    'vqa_cot': "Please first provide the reasoning process that leads to the correct answer, and then give the final answer in the format 'ANSWER: XXX'.",
    'mcq_direct': "Please answer with the option's letter from the given choices directly.",
    'vqa_direct': "Please answer with a single word or phrase directly.",
    'vqa_counting': "Please answer with a single number or phrase directly.",
    'vqa_yn': "Please answer with yes or no directly.",
    'none': '',
}


def get_choice_text(choices):
    choice_list = []
    for i, c in enumerate(choices):
        c = c if c != "" else "None"
        choice_list.append(f"{OPTIONS[i]}. {c}")
        # choice_list.append("({}) {}".format(OPTIONS[i], c))
    choice_txt = "\n".join(choice_list)
    return choice_txt


def find_number_lists_in_string(s):
    # This pattern matches brackets containing comma-separated floats
    pattern = r'\[\s*(?:\d*\.\d+|\d+)(?:\s*,\s*(?:\d*\.\d+|\d+))*\s*\]'
    matches = re.findall(pattern, s)
    # Convert string matches to actual lists of floats
    number_lists = [eval(match) for match in matches]
    return number_lists


def contains_number_list(s):
    # Pattern to match a bracketed list of comma-separated numbers
    pattern = r'\[\s*(?:\d*\.\d+|\d+)(?:\s*,\s*(?:\d*\.\d+|\d+))*\s*\]'
    # Check if there is at least one match in the string
    return bool(re.search(pattern, s))


def contains_letter_bracket(s):
    # Pattern to match a bracketed list of comma-separated numbers
    pattern = r'\(([A-Za-z])\)'
    # Check if there is at least one match in the string
    return bool(re.search(pattern, s))


def contains_image_tags(text):
    # Regular expression to match patterns like <image 1>, <image 2>, etc.
    pattern = r'<image \d+>'
    # Search the text for any match of the pattern
    match_found = bool(re.search(pattern, text))
    return match_found


def convert_choice_string2list(text):
    # 使用正则表达式匹配每行的以字母和冒号/句号开头，后跟任何内容的模式
    pattern = r"^[A-H][\.:]\s*(.+)$"
    results = re.findall(pattern, text, re.MULTILINE)
    return results


def remove_numbered_prefix(lst):
    pattern = r'^[A-Z]\.\s*'  # 匹配以大写字母加点和空格开头的模式
    result = [re.sub(pattern, '', item) for item in lst]
    return result


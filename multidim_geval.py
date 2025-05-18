from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ConversationalTestCase
from deepeval.metrics import ConversationalGEval
import fire
import json


def load_jsonl(filename):
    results = []
    with open(filename) as f:
        for line in f:
            results.append(json.loads(line))

    return results

def load_text(filename):
    with open(filename, 'r') as f:
        return f.read().strip()


def evaluate_convs(
        conversations_path: str = 'output/truncated_chats.json',
        exploration_criteria_path: str = '../es_docs/exploration_stage_static.txt',
        insight_criteria_path: str = '../es_docs/insight_stage_static.txt',
        action_criteria_path: str = '../es_docs/action_stage_static.txt'
):

    conv_data = load_jsonl(conversations_path)

    exploration_criteria = load_text(exploration_criteria_path)
    insight_criteria = load_text(insight_criteria_path)
    action_criteria = load_text(action_criteria_path)

    convo_test_cases = []

    for conv in conv_data:
        test_case_turns = []

        for i in range(1, len(conv['turns'])-1):
            test_case_turns.append(LLMTestCase(input=conv['turns'][i], actual_output=conv['turns'][i+1]))

        test_case = ConversationalTestCase(turns=test_case_turns)
        convo_test_cases.append(test_case)


    # todo: for test only take 2 samples
    convo_test_cases = convo_test_cases[:2]

    exploration_metric = ConversationalGEval(
        name="Exploration",
        criteria=exploration_criteria,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )

    insight_metric = ConversationalGEval(
        name="Insight",
        criteria=insight_criteria,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )

    evaluate(test_cases=convo_test_cases, metrics=[exploration_metric, insight_metric])


if __name__ == "__main__":
    fire.Fire(evaluate_convs)
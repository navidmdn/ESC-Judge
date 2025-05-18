import random
import numpy as np
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from random import randint, choice

basic_family_life_stage_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""You are a helpful assistant for generating detailed synthetic characteristics of people."""),
    HumanMessagePromptTemplate.from_template("""Here is an initial challenge that the person is dealing with: {challenge}\n\nFollow these steps:
 1. Assume the person is a {gender}. Randomly select a feasible age for the person. 
 2. Based on age and gender generate a list of {Nf_total} possible family or relationship statuses that the person could have (e.g. single, married, divorced, having two children etc.
 3. choose status {Nf} from the list
 4. Write a list of {No_total} occupations that the person might have
 5. choose occupation {No} from the list
 
 Compile the complete persona of the person highlighting their age, occupation, identities and their ongoing challenge and write it after "Final Persona:" """)
])

key_life_events_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""You are a helpful assistant for generating detailed synthetic life events of people."""),
    HumanMessagePromptTemplate.from_template("""Here is an initial persona for you to work with: {final_persona}\n\nFollow these steps:
 1. consider the following types of key life events: 
- Childhood Trauma or Positive Experience: Was there a significant event early in life that shaped how the character views the world? Examples could include the loss of a parent, being bullied, or a formative achievement like excelling in academics or sports.
- Family Dynamics: Consider events involving the family unit—divorce, moving to a new place, sibling rivalry, or the birth of a sibling.
- Romantic Relationships: A pivotal breakup, a deep connection with a partner, or infidelity can have long-lasting emotional consequences.
- Career Milestones or Failures: Include moments of triumph or professional failure, promotions, or the realization of a long-held career goal (or the lack thereof).
-Loss or Bereavement: The death of a loved one can profoundly affect one’s emotional responses and coping mechanisms.
-Personal Achievements: What accomplishments are important to them? It could be publishing a book, graduating from a prestigious institution, or overcoming an addiction.
 Now, write a list of {total_events} possible key life events that the person could have experienced based on the mentioned categories, 
 it shouldn't necessarily match their ongoing challenge because it is something that has happened in their past. start with "Possible key events:". 
 2. choose events {elist} from the list
 3. compile the complete list of events in this format "Key Events:" """)
])

key_life_events_prompt2 = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""You are a helpful assistant for generating detailed synthetic life events of people."""),
    HumanMessagePromptTemplate.from_template("""Here is an initial persona for you to work with: {final_persona}\n\nFollow these steps:
 1. consider the following types of key life events: 
- Childhood Trauma or Positive Experience: Was there a significant event early in life that shaped how the character views the world? Examples could include the loss of a parent, being bullied, or a formative achievement like excelling in academics or sports.
- Family Dynamics: Consider events involving the family unit—divorce, moving to a new place, sibling rivalry, or the birth of a sibling.
- Romantic Relationships: A pivotal breakup, a deep connection with a partner, or infidelity can have long-lasting emotional consequences.
- Career Milestones or Failures: Include moments of triumph or professional failure, promotions, or the realization of a long-held career goal (or the lack thereof).
-Loss or Bereavement: The death of a loved one can profoundly affect one’s emotional responses and coping mechanisms.
-Personal Achievements: What accomplishments are important to them? It could be publishing a book, graduating from a prestigious institution, or overcoming an addiction.
 Now, write a list of {total_events} possible key life events that the person could have experienced based on the mentioned categories, 
 it shouldn't necessarily match their ongoing challenge because it is something that has happened in their past. start with "Possible key events:". 
 2. choose event {K} from the list
 3. write {sub_events} possible scenarios about the chosen life event category
 4. choose event {M} from the list in this format "Key Events:" """)
])


stressors = {
    "Personal Loss & Major Life Changes": [
        "Death of a loved one",
        "Divorce or breakup of a significant relationship",
        "Family estrangement or disownment",
        "Major illness or injury (self or a loved one)",
        "Becoming a new parent",
        "Caring for an aging or ill family member",
        "Pregnancy and childbirth complications",
        "Infertility or miscarriage",
        "Losing close friends or social isolation",
        "Dealing with immigration and being away from family"
    ],
    "Identity, Discrimination & Social Challenges": [
        "Dealing with identity (being non-binary or LGBTQ+)",
        "Lack of acceptance from family or community regarding gender or sexual identity",
        "Experiencing discrimination or prejudice (race, gender, sexual orientation, disability, etc.)",
        "Workplace discrimination or harassment",
        "Identity crisis (e.g., questioning beliefs, gender, or cultural belonging)",
        "Public scandal or personal reputation damage"
    ],
    "Career & Academic Pressures": [
        "Job loss or sudden unemployment",
        "Workplace conflict or toxic work environment",
        "Career change or uncertainty about career path",
        "Burnout or chronic overworking",
        "Failure to get a promotion or recognition",
        "Struggling in school or failing courses",
        "Completing a major academic program (e.g., PhD, medical school)",
        "Job relocation to a new city or country",
        "Fear of job automation or industry decline"
    ],
    "Financial & Economic Stress": [
        "Significant debt (student loans, credit cards, etc.)",
        "Inability to pay rent or mortgage",
        "Foreclosure or eviction",
        "Sudden unexpected expenses (e.g., medical bills, car repairs)",
        "Loss of financial investments or retirement savings",
        "Poverty or living paycheck to paycheck",
        "Economic downturn affecting personal finances",
        "Supporting dependents financially (e.g., aging parents, children)",
        "Lawsuits or legal financial burdens",
        "Bankruptcy"
    ],
    "Health & Well-being": [
        "Chronic illness or disability",
        "Mental health struggles (anxiety, depression, PTSD, etc.)",
        "Sleep deprivation or chronic fatigue",
        "Major surgery or hospitalization",
        "Long-term effects of a past trauma",
        "Eating disorders or body image issues",
        "Addiction or substance abuse (self or a loved one)",
        "Side effects from medications",
        "Coping with a terminal illness"
    ],
    "Environmental & Societal Stressors": [
        "Moving to a new city or country",
        "Natural disasters (earthquakes, hurricanes, wildfires)",
        "Political instability, war, or civil unrest",
        "Dealing with crime or being a victim of violent crime",
        "Legal trouble (lawsuits, criminal charges, imprisonment)",
        "Sudden or forced lifestyle change (e.g., joining the military, religious conversion)"
    ]
}



def sample_ongoing_challenge_string(n_samples):
    return [_sample_ongoing_challenge_string() for _ in range(n_samples)]


def _sample_ongoing_challenge_string():
    total_categories = list(stressors.keys())
    sample_category = random.sample(total_categories, k=1)[0]
    sub_categories = stressors[sample_category]
    return random.sample(sub_categories, k=1)[0]



generate_full_role = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a helpful assistant for generating detailed, coherent and realistic synthetic characteristics of people.\
 Your task is to generate a system message that provides the full details of the role of a person who is seeking emotional support and has come to a counselor/therapist\
  to discuss their ongoing challenges and life stressors."""),
    HumanMessagePromptTemplate.from_template("""Use the following components and compile them into a full role description:
1. Persona: {final_persona}\n\n
2. Key life events: {key_events}\n\n
3. Behavioral traits: {behavioral_traits}\n\n
4. Ongoing challenge: {ongoing_challenges}\n\n

compile the complete role description into a detailed consistent and nuanced role playing system prompt that can be used to prompt a language model.\
 The output should consist the four mentioned categories each summerized into bullet points.
 Write the system prompt after "System Prompt:" """)
])



therapy_behavior_traits = {

    "Big Five Personality Traits": {
        "Extraversion": {
            "Introverted": "You are more reserved and may need more prompting to share thoughts and emotions.",
            "Extroverted": "You are outgoing and engages openly, easily expressing thoughts and feelings."
        },

        "Neuroticism (Emotional Stability)": {
            "Emotionally Stable": "You remain calm and composed, handling stress with resilience.",
            "Emotionally Reactive": "You experience heightened emotional responses, struggling with anxiety or mood swings."
        },

        "Conscientiousness": {
            "Disciplined": "You are goal-oriented, organized, and methodical in addressing their concerns.",
            "Impulsive": "You struggle with planning and may act on emotions without considering long-term consequences."
        },

        "Agreeableness": {
            "Empathetic": "You are warm, trusting, and open to collaboration in the helping process.",
            "Detached": "You may be skeptical, resistant, or struggle to engage emotionally in conversations."
        },

        "Openness to Experience": {
            "Curious": "You are open to new perspectives, willing to explore different solutions and reflect on emotions.",
            "Traditional": "You prefer familiar approaches, may resist change, and values structured, predictable guidance."
        }
    },

    "Cognitive Biases, Thinking Patterns, and Emotional Baseline": {
        "Cognitive Biases": {
            "Catastrophizing": "You expect the worst possible outcome in every situation.",
            "Black-and-white thinking": "You view situations as all good or all bad, with no middle ground.",
            "Overgeneralizing": "You make broad conclusions based on isolated incidents.",
            "Emotional reasoning": "You believe that their emotions reflect objective reality (e.g., feeling worthless means they are worthless)."
        },
        "Emotional Baseline": {
            "Hyper-aroused": "You are restless, easily triggered, and may have difficulty focusing due to heightened anxiety.",
            "Hypo-aroused": "You appear emotionally shut down or detached, showing little emotional engagement.",
            "Emotionally volatile": "You experience rapid emotional swings, moving between different emotional states quickly."
        }
    },


    "Response Style Toward the Therapist and Trust in the Process": {
        "Response Style": {
            "Easily reassured": "You calm down quickly with reassurance, validation, or soothing techniques.",
            "Needs logical explanation": "You respond best to structured, evidence-based interventions and logical reasoning.",
            "Resistant and defensive": "You are skeptical of the therapist, may challenge suggestions, and is resistant to intervention.",
            "Emotionally reactive": "You react strongly to perceived slights or misunderstandings, possibly becoming angry or withdrawn."
        },
        "Trust in the Process": {
            "Positive experience": "You trust the therapist and the process based on prior success.",
            "Negative experience": "You are skeptical or fearful of the process due to past negative interactions with therapists.",
            "First-time experience": "You are unfamiliar with therapy but open to exploring it, though they may be apprehensive."
        }
    },

    "Social Support Network and Coping Mechanisms": {
        "Social Support Network": {
            "Strong support": "You have a reliable network of family and friends for emotional support, which can help or hinder progress.",
            "Weak or nonexistent support": "You feel isolated and may rely heavily on the therapist for emotional regulation.",
            "Conflicted support": "You have strained relationships with key people in their life, potentially increasing stress."
        },
        "Coping Mechanisms": {
            "Adaptive coping": "You use healthy coping strategies like mindfulness, exercise, or seeking social support.",
            "Maladaptive coping": "You engage in destructive coping strategies such as substance abuse or aggression.",
            "Avoidant coping": "You avoid confronting painful issues by deflecting or minimizing the problem."
        }
    },

    "Triggers, Sensitivities, and Self-soothing Mechanisms": {
        "Triggers": {
            "Topic-specific triggers": "Certain subjects, such as family or past trauma, provoke a strong emotional response from the client.",
            "Therapist-specific triggers": "The therapist’s tone, body language, or choice of words may unintentionally set off a negative reaction.",
            "Environmental triggers": "External factors such as background noise or discomfort in the setting may distract or distress the client."
        },
        "Self-soothing Mechanisms": {
            "Rationalization": "You try to calm themselves by using logic to downplay emotional distress.",
            "Distraction": "You shift focus away from anxiety by talking about unrelated subjects or asking unrelated questions.",
            "Suppression": "You ignore or suppress emotions, which may lead to delayed or intensified emotional reactions later."
        }
    }
}

def sample_gender(n_samples):
    gids = np.random.randint(low=0, high=2, size=n_samples)
    return ['Male' if g == 1 else 'Female' for g in gids]

def sample_behavior_trait_string(n_samples):
    return [_sample_behavior_trait_string() for _ in range(n_samples)]

def _sample_behavior_trait_string():
    result = ""
    for category, cat_items in therapy_behavior_traits.items():
        result += f"- {category}\n"
        for trait, options in cat_items.items():
            option_idx = randint(0, len(options) - 1)
            chosen_option = list(options.keys())[option_idx]
            chosen_description = options[chosen_option]

            result += f"{chosen_option}: {chosen_description}\n\n"

    return result
"""
Comprehensive synthetic dataset generator.

Creates large, realistic datasets that mimic real-world mental health data:
  - Text: 1500+ Reddit-style posts across depression, anxiety, stress, and normal categories
  - Questionnaire: 2000 respondents with correlated behavioral features

These synthetic datasets allow full pipeline testing. Replace with real Kaggle
data (e.g., Reddit Depression dataset) for production results.
"""

import os
import sys
import random
import hashlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

random.seed(42)
np.random.seed(42)

# =====================================================================
#  TEXT DATASET — Reddit-style mental health posts
# =====================================================================

# Distress templates organized by subcategory for diversity
_DEPRESSION_POSTS = [
    # Hopelessness & despair
    "I feel so hopeless and I don't know what to do anymore. Every day feels the same and nothing changes no matter how hard I try.",
    "Everything feels pointless and I can't see a future for myself. I used to have dreams but now I can't even imagine tomorrow.",
    "I wake up every morning wishing I hadn't. The weight of existing is too much to bear and I don't see the point.",
    "There's this emptiness inside me that never goes away. I've tried everything but nothing fills the void.",
    "I feel like I'm just going through the motions of life without actually living. Nothing has meaning anymore.",
    "Life has lost all color and meaning. I exist but I don't really live. Each day blends into the next.",
    "I can't remember the last time I felt genuinely happy. It's like that part of me died a long time ago.",
    "No matter what I achieve it never feels like enough. The emptiness always comes back stronger than before.",
    "I've been staring at walls for hours unable to motivate myself to do anything. What's the point of trying.",
    "Sometimes I wonder if anyone would even notice if I just stopped showing up to everything.",

    # Sleep & fatigue
    "I haven't been able to sleep for days and I feel terrible. My mind won't shut off and the exhaustion is killing me.",
    "I sleep fourteen hours a day and still wake up exhausted. Getting out of bed feels like climbing a mountain.",
    "My insomnia has gotten so bad that I haven't had a proper night's sleep in weeks. I'm barely functioning.",
    "I feel exhausted all the time even though I do nothing productive. Just existing takes all my energy.",
    "I can't sleep at night because my thoughts won't stop racing. Then during the day I can barely keep my eyes open.",
    "The fatigue is overwhelming. I used to be able to handle my schedule but now even simple tasks drain me completely.",
    "I've been sleeping through my alarms and missing work. I just can't seem to care enough to get up anymore.",
    "Every night I lie awake replaying every mistake I've ever made. The lack of sleep is destroying my ability to function.",

    # Social isolation
    "I cry every night and nobody understands me. I've stopped talking to people because they just don't get it.",
    "I feel like a burden to everyone around me. They'd all be better off without me dragging them down.",
    "I've been isolating myself from friends and family for months now. I don't have the energy to pretend I'm okay.",
    "I feel completely alone even when people are around me. There's this invisible wall between me and everyone else.",
    "Nobody really knows how much I'm struggling. I put on a mask every day and it's getting heavier.",
    "I cancelled plans with friends again today. I just can't bring myself to socialize when everything hurts.",
    "My friends have stopped inviting me to things because I always say no. I want to go but I just can't make myself.",
    "I pushed away the last person who cared about me. I don't deserve their kindness and they deserve better.",
    "I deleted all my social media because seeing everyone happy made me feel worse about my own situation.",
    "I haven't left my apartment in two weeks. The thought of interacting with people fills me with dread.",

    # Self-worth
    "I feel worthless and nobody would notice if I disappeared. I contribute nothing to anyone's life.",
    "I hate looking at myself in the mirror. I see nothing but failure and disappointment staring back at me.",
    "I'm a disappointment to everyone who ever believed in me. I've wasted every opportunity I've been given.",
    "I compare myself to everyone and always come up short. I'll never be good enough no matter what I do.",
    "Every time something goes wrong I know it's because I'm fundamentally broken as a person.",
    "I feel like a fraud in everything I do. Eventually everyone will see how incompetent I really am.",
    "My self esteem is at an all time low and I genuinely hate who I have become.",
    "I can't accept compliments because I know they're just being polite. Nobody actually thinks I'm worth anything.",

    # Loss of interest
    "Nothing makes me happy anymore. I've lost all interest in the things that used to bring me joy.",
    "I used to love playing guitar but I haven't touched it in months. I used to love reading but I can't focus.",
    "I have no motivation to do anything productive. Even my favorite hobbies feel like pointless chores now.",
    "I quit my hobbies one by one because none of them bring me any satisfaction anymore. I just sit and stare.",
    "Food doesn't taste good anymore. Music doesn't move me. Nothing triggers any positive emotion in me.",
    "I used to be passionate about so many things but now I feel completely numb to everything around me.",

    # Cognitive symptoms
    "I can't concentrate on anything my mind feels foggy all the time. I read the same paragraph five times.",
    "My memory has gotten so bad lately. I forget conversations I had yesterday and lose track of what I'm doing.",
    "I can't make decisions anymore. Even choosing what to eat feels overwhelming and paralyzing.",
    "My thoughts are so scattered that I can't complete a single task. I start things and immediately forget why.",
    "Brain fog is ruining my work performance. I used to be sharp but now I can barely string sentences together.",

    # Physical symptoms
    "I've lost fifteen pounds in the last month because I can't bring myself to eat. Food makes me nauseous.",
    "My body aches constantly and the doctors say nothing is physically wrong. The pain is real though.",
    "I've been having headaches every day for weeks. The stress and sadness are literally making me sick.",
    "I can't eat or sleep properly anymore. My body is falling apart along with my mental state.",
    "My chest feels tight all the time like there's a weight pressing down on me. It's hard to breathe sometimes.",

    # Crying & emotional pain
    "I have been crying uncontrollably for no specific reason. The tears just come and I can't stop them.",
    "I feel numb inside as if nothing matters anymore. I wish I could cry but I can't even feel that anymore.",
    "The sadness won't go away no matter what I try. It's like a permanent cloud hanging over everything.",
    "I feel like I'm drowning in my own emotions and there's no one to pull me out.",
    "I keep having breakdowns at random times. In the shower at work while driving. I can't control it.",
    "The emotional pain is so intense it becomes physical. My chest hurts and my stomach is always in knots.",

    # Anxiety overlap
    "I've been having panic attacks almost every day. My heart races and I can't breathe and I think I'm dying.",
    "My anxiety is through the roof and I can't function normally. Every small task feels like a threat.",
    "I can't stop worrying about everything that could go wrong. My mind creates disasters that haven't happened.",
    "I feel trapped in my own thoughts and can't escape the cycle of negative thinking.",
    "I keep reliving past traumas and it breaks me every time. The memories won't leave me alone.",
    "I am afraid of the future and feel paralyzed by uncertainty. I can't plan anything because what's the point.",
    "I feel disconnected from reality and from myself. Sometimes I don't even recognize who I am anymore.",

    # Work & academic stress leading to depression
    "I can't handle the pressure from work anymore. I used to love my job but now it's destroying me.",
    "I failed my exams again and I feel like giving up on everything. I'm clearly not smart enough for this.",
    "My boss criticized my work today and I wanted to disappear. I spent the rest of the day fighting back tears.",
    "College is crushing me. The assignments the deadlines the expectations. I'm drowning and nobody sees it.",
    "I got fired today and I feel like my life is over. I have nothing left and no reason to keep going.",
    "I've been putting in sixty hour weeks and I'm completely burned out. I have nothing left to give.",

    # Relationship issues
    "My partner left me and I don't know how to exist without them. They were the only reason I kept going.",
    "Nobody truly cares about how I'm doing. When they ask how are you they don't want the real answer.",
    "I feel like nobody would notice if I just disappeared from everyone's life. I'm invisible.",
    "Every relationship I've ever had has ended because I'm too broken. I push people away because I'm terrified.",
    "I just found out my best friend has been talking behind my back. The one person I trusted betrayed me.",

    # Existential
    "Life feels meaningless and empty to me lately. I go through each day wondering what the point of it all is.",
    "I keep thinking about giving up on everything. Not in a harmful way just stopping trying altogether.",
    "What is the point of trying when everything falls apart eventually. Nothing I build ever lasts.",
    "I feel suffocated by my responsibilities and expectations. Everyone wants something from me and I have nothing left.",
    "Everything overwhelms me and I want to shut the world out. I just want silence and peace.",
    "I am constantly questioning whether any of this matters. We're all just going to be forgotten anyway.",

    # Long-form Reddit style
    "I don't even know where to start. I've been struggling with depression for years and lately it's gotten so much worse. I can't hold down a job I've pushed away everyone who cared about me and I spend most days just lying in bed staring at the ceiling. I know I should get help but I don't have the energy or money for therapy. I feel stuck in this endless loop of suffering and I don't see a way out. Has anyone else been through this and actually made it to the other side.",
    "Today marks three months since I last felt anything resembling happiness. I go through my day on autopilot doing the bare minimum to survive. My apartment is a mess I haven't done laundry in weeks and I've been eating nothing but instant noodles. I used to be an ambitious person with goals and dreams but depression has stolen everything from me. I'm posting here because I have literally no one else to talk to.",
    "I had another breakdown at work today. My manager asked me a simple question and I just froze. Couldn't think couldn't speak couldn't move. Everyone was staring at me and I eventually just walked out. I'm probably going to get fired which will make everything worse. The cycle never ends. I'm so tired of fighting a battle I can never win.",
    "My therapist told me to write down my feelings so here goes. I feel like I'm watching my life from outside my body. Nothing feels real anymore. I go through the motions but there's no one home inside. The medication helps with the sharp edges but it's replaced the pain with numbness which is somehow worse. At least when I was in pain I could feel something.",
    "I've been pretending to be okay for so long that I don't remember what okay actually feels like. Everyone thinks I'm fine because I've gotten so good at wearing the mask. But underneath it I'm falling apart piece by piece. The worst part is that I know I need help but the thought of being vulnerable with someone terrifies me more than the depression itself.",
]

_NORMAL_POSTS = [
    # Positive daily life
    "Had a wonderful day at work today. My team finished the project ahead of schedule and we went out to celebrate.",
    "Spent the entire afternoon at the park with my dog. The weather was perfect and I feel so refreshed and happy.",
    "I'm feeling grateful for my family and their constant support. Had a great family dinner tonight.",
    "Just finished an amazing workout at the gym. Hit a new personal record on squats and I'm feeling proud.",
    "Looking forward to the weekend plans with friends. We're going hiking at the national park.",
    "Had a meaningful conversation with my best friend today. We talked for hours and it made my whole week.",
    "I'm making great progress on my personal goals. Started learning piano last month and I can already play a few songs.",
    "Enjoyed cooking a new recipe tonight. It turned out way better than expected and my family loved it.",
    "Feeling calm and peaceful after my morning meditation session. Twenty minutes of mindfulness really sets the tone for the day.",
    "Things are going really well at work. Got positive feedback from my manager and a potential promotion coming up.",

    # Optimism & growth
    "I'm feeling optimistic about the future and excited about my plans for the new year.",
    "Had a great night's sleep and woke up feeling completely refreshed. Today is going to be a good day.",
    "I'm really proud of what I accomplished this week. Finished three major tasks and still had time for myself.",
    "Enjoying my hobbies more than ever. Started painting again after years and it brings me so much joy.",
    "Met a new friend at a networking event today. We had an inspiring conversation about our shared interests.",
    "The weather was beautiful today so I went for a long walk along the river. Nature always helps me reset.",
    "I'm feeling balanced and content with my life right now. Not everything is perfect but I'm at peace.",
    "Learned something new at the workshop today and it was genuinely exciting. I love growing as a person.",
    "I feel supported and loved by the people around me. I'm lucky to have such a great support system.",
    "My confidence has been growing steadily. I spoke up in a meeting today and everyone really listened.",

    # Daily activities
    "Cooked a healthy meal from scratch today and felt really accomplished. Meal prepping for the whole week.",
    "Had a productive study session at the library. Got through two chapters and actually understood everything.",
    "Helped a stranger carry their groceries today. Small acts of kindness always make me feel warm inside.",
    "Managed my time really well today and finished all my tasks before dinner. Feeling efficient.",
    "I feel strong and capable of handling whatever challenges come my way. Bring it on.",
    "Attended a community volunteer event this morning. Giving back always puts things in perspective.",
    "Completed a project I've been working on for three weeks. The sense of accomplishment is incredible.",
    "Had a creative breakthrough while working on my art today. Everything just clicked into place.",
    "Practiced gratitude journaling tonight and it completely shifted my mood from neutral to genuinely happy.",
    "I feel safe and secure in my current living situation. Finally moved into a place that feels like home.",

    # Relationships & social
    "Spent quality time with my partner today. We went to a museum and then had dinner at our favorite restaurant.",
    "My friend group organized a surprise party for my birthday. I feel so blessed to have people who care.",
    "Had a deep conversation with my mom today. We talked about life and she gave me some amazing advice.",
    "Went to a book club meeting tonight and had a fantastic discussion. Love the intellectual stimulation.",
    "My neighbor brought over homemade cookies today just because. The world has such kind people in it.",
    "Reconnected with an old college friend over video call. It was like no time had passed at all.",
    "I set healthy boundaries with a difficult coworker today and it felt empowering and freeing.",
    "Got together with friends for game night. We laughed so hard my cheeks hurt. Best evening in weeks.",

    # Self-improvement
    "Ran my first five kilometers without stopping today. Three months ago I couldn't run for two minutes.",
    "Finally finished reading that book I started. It gave me a completely new perspective on my career.",
    "I've been consistently meditating for thirty days now and the difference in my mental clarity is remarkable.",
    "Started a new online course in data science. The material is challenging but I love the learning process.",
    "Woke up early and watched the sunrise with my coffee. These quiet morning moments are pure gold.",
    "I successfully managed a stressful deadline at work without losing my cool. Growth feels good.",
    "Reflected on how much I've grown in the past year. I'm genuinely proud of the person I'm becoming.",
    "Spent time in nature today and it completely refreshed my mind. Went on a forest trail with wildflowers everywhere.",
    "I organized my workspace today and the clarity it brought to my thinking was immediate.",
    "Tried a new sport today and even though I was terrible at it I had the most fun I've had in months.",

    # Contentment
    "Nothing extraordinary happened today and that's perfectly fine. A calm ordinary day is a gift.",
    "I appreciate the simple things more now. A warm cup of tea a good book a quiet evening. Life is good.",
    "Feeling grateful for my health today. Took a long bike ride and felt the wind on my face. Freedom.",
    "I don't need everything to be perfect to be happy. Today was messy and imperfect and I loved it.",
    "Sat in my garden this evening and just listened to the birds. Pure peace and contentment.",
    "I'm exactly where I need to be right now. Not where I want to be yet but making steady progress.",

    # Long form positive
    "Today was one of those days that reminds you why life is worth living. Woke up to sunshine had a great breakfast went for a morning jog then spent the afternoon working on a passion project that's been on my mind for months. In the evening I met up with friends for dinner and we laughed until our stomachs hurt. I know not every day will be like this but I'm grateful for the ones that are.",
    "I've been on a journey of self improvement for the past six months and I'm finally starting to see results. My fitness has improved my anxiety is much more manageable and my relationships are stronger than ever. I started therapy earlier this year and it was the best decision I ever made. To anyone considering it you deserve to invest in yourself.",
    "Big milestone today. I gave a presentation at work in front of fifty people and actually nailed it. Six months ago the thought of public speaking would have made me physically sick but I've been working on my confidence through practice and preparation. The feeling of overcoming a fear is absolutely incredible.",
]

# Augmentation: generate paraphrased variants
def _augment_texts(texts: list, label: int, target_count: int) -> list:
    """Generate augmented text samples through template mixing and variation."""
    records = []

    for text in texts:
        records.append({"text": text, "label": label})

    # Word-level augmentations
    intensifiers = ["really", "truly", "absolutely", "completely", "totally", "genuinely", "honestly"]
    connectors = ["and", "also", "plus", "moreover", "additionally"]

    while len(records) < target_count:
        base = random.choice(texts)
        aug_type = random.choice(["shuffle_sentences", "add_intensifier", "combine", "truncate", "rephrase_start"])

        if aug_type == "shuffle_sentences":
            sentences = [s.strip() for s in base.replace(". ", ".\n").split("\n") if s.strip()]
            if len(sentences) > 2:
                random.shuffle(sentences)
                new_text = ". ".join(sentences)
                records.append({"text": new_text, "label": label})

        elif aug_type == "add_intensifier":
            words = base.split()
            if len(words) > 5:
                pos = random.randint(1, len(words) - 2)
                words.insert(pos, random.choice(intensifiers))
                records.append({"text": " ".join(words), "label": label})

        elif aug_type == "combine":
            other = random.choice(texts)
            if other != base:
                sentences_a = base.split(". ")
                sentences_b = other.split(". ")
                if len(sentences_a) > 1 and len(sentences_b) > 1:
                    combined = ". ".join(sentences_a[:2] + [random.choice(connectors)] + sentences_b[:2])
                    records.append({"text": combined, "label": label})

        elif aug_type == "truncate":
            sentences = base.split(". ")
            if len(sentences) > 2:
                cut = random.randint(1, len(sentences) - 1)
                records.append({"text": ". ".join(sentences[:cut]), "label": label})

        elif aug_type == "rephrase_start":
            starters_distress = ["Honestly ", "I just need to vent. ", "Can anyone relate? ", "I feel like ",
                                 "Does anyone else feel this way? ", "I hate admitting this but ",
                                 "Throwaway account because ", "I don't know who to tell but "]
            starters_normal = ["Just wanted to share. ", "Had a great day! ", "Feeling thankful. ",
                              "Good vibes today. ", "Wanted to spread some positivity. ", "Life update: "]
            starters = starters_distress if label == 1 else starters_normal
            new_text = random.choice(starters) + base[0].lower() + base[1:]
            records.append({"text": new_text, "label": label})

    return records[:target_count]


def generate_text_dataset(distress_count: int = 800, normal_count: int = 800):
    """Create a comprehensive text-depression dataset with augmentation."""
    print("[DataGen] Building text dataset with augmentation...")

    distress_records = _augment_texts(_DEPRESSION_POSTS, label=1, target_count=distress_count)
    normal_records = _augment_texts(_NORMAL_POSTS, label=0, target_count=normal_count)

    all_records = distress_records + normal_records
    df = pd.DataFrame(all_records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Remove exact duplicates
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    out_path = config.TEXT_CONFIG["raw_csv"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[DataGen] Text dataset: {out_path} ({len(df)} samples, "
          f"{(df['label']==1).sum()} distress / {(df['label']==0).sum()} normal)")
    return df


def generate_questionnaire_dataset(n_samples: int = 2000):
    """
    Create a realistic questionnaire dataset with correlated features.

    Simulates respondents with:
      - Correlated behavioral features (sleep affects energy, mood affects interest, etc.)
      - Realistic distributions per risk group
      - Noise to prevent trivial separation
    """
    print("[DataGen] Building questionnaire dataset with correlated features...")
    feature_names = [item["id"] for item in config.QUESTIONNAIRE_ITEMS]
    records = []

    for _ in range(n_samples):
        risk = random.choices([0, 1, 2], weights=[0.35, 0.35, 0.30])[0]

        # Base feature generation with inter-feature correlations
        if risk == 0:  # Low risk
            base_wellbeing = np.random.normal(7.5, 1.0)
            sleep = np.clip(np.random.normal(base_wellbeing + 0.5, 0.8), 1, 10)
            appetite = np.clip(np.random.normal(base_wellbeing, 1.0), 1, 10)
            energy = np.clip(np.random.normal(sleep * 0.6 + 3, 0.8), 1, 10)
            social = np.clip(np.random.normal(base_wellbeing - 0.5, 1.2), 1, 10)
            concentration = np.clip(np.random.normal(energy * 0.5 + 3, 0.9), 1, 10)
            mood = np.clip(np.random.normal(base_wellbeing + 0.3, 0.7), 1, 10)
            stress = np.clip(np.random.normal(3.0, 1.5), 1, 10)
            physical = np.clip(np.random.normal(energy * 0.4 + 4, 1.0), 1, 10)
            interest = np.clip(np.random.normal(mood * 0.6 + 3, 0.8), 1, 10)
            self_worth = np.clip(np.random.normal(base_wellbeing, 0.9), 1, 10)

        elif risk == 1:  # Moderate risk
            base_wellbeing = np.random.normal(5.0, 1.2)
            sleep = np.clip(np.random.normal(base_wellbeing, 1.2), 1, 10)
            appetite = np.clip(np.random.normal(base_wellbeing - 0.3, 1.3), 1, 10)
            energy = np.clip(np.random.normal(sleep * 0.5 + 2, 1.0), 1, 10)
            social = np.clip(np.random.normal(base_wellbeing - 1, 1.5), 1, 10)
            concentration = np.clip(np.random.normal(energy * 0.4 + 2, 1.2), 1, 10)
            mood = np.clip(np.random.normal(base_wellbeing - 0.5, 1.2), 1, 10)
            stress = np.clip(np.random.normal(6.0, 1.5), 1, 10)
            physical = np.clip(np.random.normal(energy * 0.3 + 2, 1.3), 1, 10)
            interest = np.clip(np.random.normal(mood * 0.5 + 2, 1.2), 1, 10)
            self_worth = np.clip(np.random.normal(base_wellbeing - 0.5, 1.3), 1, 10)

        else:  # High risk
            base_wellbeing = np.random.normal(2.8, 1.0)
            sleep = np.clip(np.random.normal(base_wellbeing - 0.3, 1.0), 1, 10)
            appetite = np.clip(np.random.normal(base_wellbeing - 0.5, 1.2), 1, 10)
            energy = np.clip(np.random.normal(sleep * 0.4 + 1, 0.9), 1, 10)
            social = np.clip(np.random.normal(base_wellbeing - 1, 1.0), 1, 10)
            concentration = np.clip(np.random.normal(energy * 0.3 + 1, 1.0), 1, 10)
            mood = np.clip(np.random.normal(base_wellbeing - 0.5, 1.0), 1, 10)
            stress = np.clip(np.random.normal(8.0, 1.0), 1, 10)
            physical = np.clip(np.random.normal(energy * 0.3 + 1, 1.0), 1, 10)
            interest = np.clip(np.random.normal(mood * 0.4 + 1, 1.0), 1, 10)
            self_worth = np.clip(np.random.normal(base_wellbeing - 1, 1.0), 1, 10)

        vals = {
            "sleep_quality": round(sleep, 1),
            "appetite": round(appetite, 1),
            "energy_level": round(energy, 1),
            "social_interaction": round(social, 1),
            "concentration": round(concentration, 1),
            "mood": round(mood, 1),
            "stress_level": round(stress, 1),
            "physical_activity": round(physical, 1),
            "interest_in_activities": round(interest, 1),
            "self_worth": round(self_worth, 1),
            "risk_level": risk,
        }
        records.append(vals)

    df = pd.DataFrame(records)
    out_path = config.QUESTIONNAIRE_CONFIG["raw_csv"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[DataGen] Questionnaire dataset: {out_path} ({len(df)} samples)")
    print(f"          Distribution: {dict(df['risk_level'].value_counts().sort_index())}")
    return df


def generate_audio_feature_dataset(n_samples: int = 1200):
    """
    Generate synthetic audio feature vectors that mimic MFCC + spectral features
    extracted from emotional speech.

    Each sample is a 34-dim feature vector labeled with an emotion.
    Feature distributions are based on published acoustic correlates of emotion:
      - Sad/Fear: lower energy, flatter spectrum, lower centroid
      - Happy/Neutral: higher energy, brighter spectrum
      - Angry: high energy, high ZCR
    """
    print("[DataGen] Building synthetic audio feature dataset...")
    from src.data.audio_loader import FEATURE_DIM

    emotion_profiles = {
        "happy":   {"mfcc_base": 5.0,  "mfcc_std": 3.0, "energy": 0.08, "zcr": 0.07, "centroid": 2500, "rolloff": 4000},
        "neutral": {"mfcc_base": 0.0,  "mfcc_std": 2.0, "energy": 0.05, "zcr": 0.05, "centroid": 2000, "rolloff": 3500},
        "calm":    {"mfcc_base": -2.0, "mfcc_std": 1.5, "energy": 0.03, "zcr": 0.04, "centroid": 1800, "rolloff": 3000},
        "sad":     {"mfcc_base": -5.0, "mfcc_std": 1.8, "energy": 0.02, "zcr": 0.03, "centroid": 1500, "rolloff": 2500},
        "angry":   {"mfcc_base": 3.0,  "mfcc_std": 4.0, "energy": 0.12, "zcr": 0.10, "centroid": 3000, "rolloff": 5000},
        "fear":    {"mfcc_base": -3.0, "mfcc_std": 3.5, "energy": 0.06, "zcr": 0.08, "centroid": 2200, "rolloff": 3800},
    }

    records = []
    emotions = list(emotion_profiles.keys())
    samples_per_emotion = n_samples // len(emotions)

    for emotion in emotions:
        prof = emotion_profiles[emotion]
        for _ in range(samples_per_emotion):
            feats = []
            # 13 MFCC means
            for i in range(13):
                scale = 10.0 if i == 0 else 3.0
                feats.append(np.random.normal(prof["mfcc_base"] * (1 if i == 0 else 0.3), scale))
            # 13 MFCC stds
            for i in range(13):
                feats.append(abs(np.random.normal(prof["mfcc_std"], 1.0)))
            # RMS energy
            feats.append(abs(np.random.normal(prof["energy"], prof["energy"] * 0.3)))
            # ZCR
            feats.append(abs(np.random.normal(prof["zcr"], prof["zcr"] * 0.25)))
            # Spectral centroid
            feats.append(abs(np.random.normal(prof["centroid"], prof["centroid"] * 0.15)))
            # Spectral rolloff
            feats.append(abs(np.random.normal(prof["rolloff"], prof["rolloff"] * 0.12)))
            # 5 spectral contrast bands
            for b in range(5):
                feats.append(abs(np.random.normal(20 + b * 3, 5)))

            record = {f"feat_{i}": round(feats[i], 6) for i in range(len(feats))}
            record["emotion"] = emotion
            records.append(record)

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = config.AUDIO_CONFIG["processed_csv"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[DataGen] Audio feature dataset: {out_path} ({len(df)} samples)")
    print(f"          Emotions: {dict(df['emotion'].value_counts().sort_index())}")
    return df


if __name__ == "__main__":
    generate_text_dataset()
    generate_questionnaire_dataset()
    generate_audio_feature_dataset()
    print("\nAll datasets generated. Run 'python train_all.py' next.")

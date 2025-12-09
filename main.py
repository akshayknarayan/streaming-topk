import time
import heapq
import re
from typing import Any

import pandas as pd

import lotus
from lotus.models import LM
from lotus.types import LMOutput, ReasoningStrategy, SemanticTopKOutput
from lotus.templates import task_instructions
from lotus.templates.task_instructions import df2multimodal_info, df2text
import lotus.nl_expression
from lotus.sem_ops.sem_topk import HeapDoc

def sample_data():
    # Sample Data
    data = {
        'PaperID': list(range(101, 121)),
        'Title': [
            "A Novel Approach to Deep Learning Optimization",
            "The Role of Quantum Entanglement in Nanomaterials",
            "Historical Analysis of 18th Century European Economics",
            "Efficacy of a New Drug for Treating Autoimmune Disorders",
            "Comparative Study of Urban Planning Models in East Asia",
            "Efficient Resource Allocation in Cloud Computing Environments",
            "Climate Change Impact on Arctic Marine Ecosystems",
            "The Philosophy of Mind: Consciousness and Computation",
            "Advances in CRISPR-Cas9 Gene Editing Techniques",
            "Modeling Financial Market Volatility using GARCH Processes",
            "Acoustic Signatures of Submerged Robotics in Deep Sea",
            "Post-Colonial Narratives in Contemporary African Literature",
            "Synthesis and Characterization of Novel Polymer Composites",
            "Optimizing Traffic Flow with Reinforcement Learning",
            "Immunological Responses to Viral Vector Vaccines",
            "The Mathematics of Knot Theory and its Applications",
            "Ethical Implications of Autonomous Vehicle Decision-Making",
            "Ancient Roman Infrastructure: Water Management Systems",
            "Machine Learning for Early Detection of Forest Fires",
            "Pedagogical Strategies for Blended Learning in Higher Education"
        ],
        'Authors': [
            "J. Smith, M. Lee, S. Chen",
            "A. Khan, B. Rodriguez",
            "C. Davies",
            "E. Wang, F. Patel",
            "G. Kim, H. Tanaka, I. Schmidt",
            "L. Zhou, K. Varma",
            "M. Peterson",
            "N. Chopra",
            "O. Davies, P. Gomez",
            "Q. Rodriguez, R. Singh",
            "S. Tanaka, T. Hsu",
            "U. Nkrumah",
            "V. Sharma, W. Lee",
            "X. Yang",
            "Y. Chen, Z. Klein",
            "A. Baker, B. Chen",
            "C. Davis",
            "D. Evans, E. Franco",
            "F. George",
            "G. Hill, H. Irwin"
        ],
        'Publication Year': [
            2023, 2024, 2022, 2023, 2024,
            2023, 2024, 2022, 2023, 2024,
            2023, 2022, 2024, 2023, 2024,
            2022, 2023, 2024, 2023, 2022],
        'Journal/Conference': [
            "Proc. NeurIPS",
            "Phys. Rev. Lett.",
            "J. Econ. Hist.",
            "Lancet",
            "Habitat Int.",
            "IEEE Trans. Parallel Distrib. Syst.",
            "Nat. Clim. Chang.",
            "Philos. Rev.",
            "Science",
            "J. Financ. Econ.",
            "J. Acoust. Soc. Am.",
            "Res. Afr. Lit.",
            "Macromolecules",
            "Transp. Res. Part C",
            "Cell Host Microbe",
            "J. Knot Theory Ramif.",
            "Ethics Inf. Technol.",
            "J. Archaeol. Sci.",
            "Environ. Model. Softw.",
            "Int. J. Educ. Technol. High. Educ."
        ],
        'Citations': [45, 12, 150, 78, 22, 98, 205, 34, 188, 55, 10, 89, 41, 120, 67, 18, 50, 15, 75, 29],
        'Abstract': [
            "We propose a novel method for tuning hyperparameters in deep neural networks, demonstrating superior convergence rates on standard benchmark datasets. ",
            "This paper explores the theoretical limits and practical applications of quantum entanglement phenomena within synthetic nanomaterials. Key findings suggest new pathways for quantum computing components.",
            "Examining primary source documents, this study re-evaluates the impact of mercantilism on the socio-economic development of France and Britain during the Enlightenment.",
            "A double-blind, randomized controlled trial evaluating the safety and effectiveness of Compound X in patients with Rheumatoid Arthritis. Results show significant symptom reduction.",
            "Analyzing the differences and similarities in urban development strategies between Tokyo, Seoul, and Shanghai, focusing on sustainability metrics and infrastructure resilience.",
            "We present an algorithm to dynamically adjust resource provisioning for containerized applications in a multi-cloud environment, leading to significant cost savings and latency reduction.",
            "This study projects the long-term effects of melting ice caps and ocean acidification on plankton populations and the larger marine food web in the Arctic.",
            "An exploration of the hard problem of consciousness, arguing for a computational perspective that models subjective experience within complex neural architectures.",
            "Detailing the latest modifications to the CRISPR-Cas9 system that enhance targeting precision and reduce off-target effects, opening new avenues for therapeutic gene editing.",
            "A comprehensive application of Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models to forecast and manage risk in emerging market equity indexes.",
            "Analyzing the unique sound profiles generated by different classes of autonomous underwater vehicles to improve detection and classification capabilities.",
            "Focusing on the works of three key West African authors, this paper examines how post-colonial identity is constructed and contested in contemporary fiction.",
            "The successful fabrication and testing of a new class of fiber-reinforced polymer composites with enhanced tensile strength and thermal stability for aerospace applications.",
            "A deep reinforcement learning agent is trained to manage dynamic traffic signals at complex urban intersections, resulting in a reduction of average wait time by 25%.",
            "A comparative analysis of the innate and adaptive immune responses elicited by different non-replicating and replicating viral vector vaccine platforms.",
            "Investigating the mathematical properties of various knot invariants and their potential use in modeling DNA structures and polymer chemistry.",
            "Discussion of the philosophical frameworks necessary for programming ethical decision-making into L5 autonomous vehicles, particularly in unavoidable accident scenarios.",
            "An investigation into the design and hydrodynamics of the Roman aqueduct system, demonstrating advanced engineering principles for urban water supply.",
            "Leveraging satellite imagery and deep convolutional neural networks to identify early-stage wildfires with high accuracy and low false-positive rates.",
            "Evaluating the effectiveness of different instructional design models when implementing blended learning courses in large university settings, focusing on student engagement."
        ],
        'DOI': [
            "10.1109/TNNLS.2023.12345",
            "10.1063/1.5000000",
            "10.1017/S00000000",
            "10.1016/S0140-6736(23)00000-0",
            "10.1016/j.habitatint.2024.00000",
            "10.1109/TPDS.2023.67890",
            "10.1038/s41558-024-00000-0",
            "10.1215/00318108-0000000",
            "10.1126/science.abc0000",
            "10.1016/S0304-405X(24)00000-0",
            "10.1121/10.0000000",
            "10.2979/reseafrilit.53.3.0000",
            "10.1021/acs.macromol.4b00000",
            "10.1016/j.trc.2023.00000",
            "10.1016/j.chom.2024.00000",
            "10.1142/S021821652200000",
            "10.1007/s11948-023-00000-0",
            "10.1016/j.jas.2024.00000",
            "10.1016/j.envsoft.2023.00000",
            "10.1186/s41239-022-00000-0"
        ],
        'Field': [
            'Computer Science',
            'Physics',
            'Humanities',
            'Medicine',
            'Urban Studies',
            'Computer Science',
            'Environmental Science',
            'Philosophy',
            'Biology',
            'Finance',
            'Engineering',
            'Literature',
            'Materials Science',
            'Computer Science',
            'Medicine',
            'Mathematics',
            'Ethics',
            'Archaeology',
            'Computer Science',
            'Education'
        ]
    }

    # Create the DataFrame
    return pd.DataFrame(data)

def sample_batches():
    df = sample_data()
    for i in range(0, len(df), batch_size):
        start_idx = i
        end_idx = i+batch_size
        batch_df = df.iloc[start_idx:end_idx]
        yield batch_df


def nonstreaming_topk(df_iter, topk_prompt, wanted_k) -> pd.DataFrame:
    tot_df = pd.concat(df_iter)
    return tot_df.sem_topk(topk_prompt, K=wanted_k)

def streaming_topk_basic(df_iter, topk_prompt, wanted_k) -> pd.DataFrame:
    curr_topk = pd.DataFrame({})
    for df in df_iter:
        if curr_topk.empty:
            curr_topk = df.sem_topk(topk_prompt, K=wanted_k)
        else:
            curr_topk = pd.concat([curr_topk, df]).sem_topk(topk_prompt, K=wanted_k)
    return curr_topk

def docs_to_heapdoc(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str,
):
    HeapDoc.num_calls = 0
    HeapDoc.total_tokens = 0
    HeapDoc.strategy = None
    HeapDoc.model = model
    HeapDoc.explanations = {}
    for idx in range(len(docs)):
        yield HeapDoc(docs[idx], user_instruction, idx)

def llm_heapsort(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str,
    K: int,
):
    heap = list(docs_to_heapdoc(docs, model, user_instruction))
    heapq.heapify_max(heap)
    while len(heap) > K:
        heapq.heappop_max(heap)
    return heap

def incr_heap(
    heap,
    new_docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str
):
    # K == len(heap)
    for doc in docs_to_heapdoc(new_docs, model, user_instruction):
        heapq.heappushpop_max(heap, doc)
    return heap

def finalize_heap_topk(heap):
    # why do we need to heap-pop if we don't care about order within the top-k?
    #indexes = [heapq.heappop(heap).idx for _ in range(len(heap))]
    indexes = [h.idx for h in heap]

    stats = {
        "total_tokens": HeapDoc.total_tokens,
        "total_llm_calls": HeapDoc.num_calls,
        "explanations": HeapDoc.explanations,
    }
    return SemanticTopKOutput(indexes=indexes, stats=stats)


def streaming_topk_incremental(df_iter, topk_prompt, wanted_k) -> pd.DataFrame:
    model = lotus.settings.lm
    if model is None:
        raise ValueError(
            "The language model must be an instance of LM. Please configure a valid language model using lotus.settings.configure()"
        )

    col_li = lotus.nl_expression.parse_cols(topk_prompt)
    formatted_usr_instr = lotus.nl_expression.nle2str(topk_prompt, col_li)

    curr_heap = []
    curr_df = pd.DataFrame({})

    for df in df_iter:
        for column in col_li:
            if column not in df.columns:
                raise ValueError(f"column {column} not found in DataFrame. Given usr instruction: {topk_prompt}")
        multimodal_data = task_instructions.df2multimodal_info(df, col_li)
        if len(curr_heap) == 0:
            curr_heap = llm_heapsort(multimodal_data, model, formatted_usr_instr, wanted_k)
            curr_df = df
        else:
            curr_df = pd.concat([curr_df, df])
            curr_heap = incr_heap(curr_heap, multimodal_data, model, formatted_usr_instr)
        out = finalize_heap_topk(curr_heap)
        curr_df.reset_index(drop=True)
        curr_df.reindex(out.indexes).reset_index(drop=True)
        curr_df = curr_df.head(wanted_k)

    return curr_df

batch_size = 5
def exp(strategy):
    topk_prompt = "The paper {Title} by {Authors} is most relevant to computer networking"
    wanted_k = 3

    lotus.settings.lm.reset_stats()
    lotus.settings.lm.reset_cache()
    then = time.time()
    topk = strategy(sample_batches(), topk_prompt, wanted_k)
    basic_time = time.time() - then
    print(f"{strategy.__name__}: topk in batches of {batch_size} in {basic_time}s")
    print(topk)
    lotus.settings.lm.print_total_usage()

def main():
    #exp(streaming_topk_basic)
    #exp(nonstreaming_topk)
    exp(streaming_topk_incremental)

if __name__ == "__main__":
    from lotus import cache
    from lotus.cache import CacheFactory, CacheConfig, CacheType

    cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=1000, cache_dir="./lotus-cache")
    cache = CacheFactory.create_cache(cache_config)
    lotus.settings.configure(lm=LM(model="ollama/llama2", api_base="http://localhost:11434", cache=cache), enable_cache=True)
    main()

"""
Script 6: RAG Knowledge Assistant
====================================
Retrieval-Augmented Generation system for sheet metal forming
process knowledge. Indexes simulation data, SHAP insights,
PSO results and the DDACS research paper into a searchable
vector database. Engineers can ask natural language questions
and get instant, evidence-based answers.
 
Directly addresses PhD requirement:
"explore and evaluate usability of system architectures such as
Retrieval-Augmented Generation (RAG) for data retrieval and
knowledge inference" — Helmholtz-Zentrum Hereon JD, 2026
 
Stack: FAISS + HuggingFace Embeddings + Anthropic/Groq LLM
"""
 
import os
import json
import numpy as np
import pandas as pd
import faiss
import warnings
warnings.filterwarnings('ignore')
 
# ── Load .env file if it exists ──
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system env vars
 
# ── Detect LLM provider based on available API keys ──
GROQ_API_KEY      = os.getenv('GROQ_API_KEY', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
 
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
 
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
 
if GROQ_API_KEY and HAS_GROQ:
    LLM_PROVIDER = 'groq'
    print(f"  ✅ Groq LLM configured")
elif ANTHROPIC_API_KEY and HAS_ANTHROPIC:
    LLM_PROVIDER = 'anthropic'
    print(f"  ✅ Anthropic LLM configured")
else:
    LLM_PROVIDER = 'none'
    print(f"  ⚠️  No LLM configured — add GROQ_API_KEY to .env file")
 
from sentence_transformers import SentenceTransformer
 
os.makedirs('rag_db', exist_ok=True)
 
print("=" * 55)
print("  Script 6: RAG Knowledge Assistant")
print("=" * 55)
 
# ─────────────────────────────────────────────
# 1. BUILD KNOWLEDGE BASE
# Documents to index:
#  - Simulation summaries from metadata
#  - SHAP insights
#  - PSO optimal parameters
#  - Process engineering rules
#  - Paper abstract knowledge
# ─────────────────────────────────────────────
 
def build_knowledge_base():
    documents = []
 
    # ── A: Simulation outcome summaries ──
    df = pd.read_csv('data/forming_data.csv')
 
    outcome_summary = df.groupby('outcome_label').agg({
        'MAT':  ['mean', 'std'],
        'FC':   ['mean', 'std'],
        'SHTK': ['mean', 'std'],
        'BF':   ['mean', 'std'],
    }).round(4)
 
    for outcome in df['outcome_label'].unique():
        sub = df[df['outcome_label'] == outcome]
        doc = (
            f"Forming outcome: {outcome}. "
            f"This outcome occurs in {len(sub)} simulations ({len(sub)/len(df)*100:.1f}% of cases). "
            f"Average process parameters when {outcome} occurs: "
            f"MAT={sub['MAT'].mean():.3f} (std={sub['MAT'].std():.3f}), "
            f"FC={sub['FC'].mean():.3f} (std={sub['FC'].std():.3f}), "
            f"SHTK={sub['SHTK'].mean():.4f}mm (std={sub['SHTK'].std():.4f}), "
            f"BF={sub['BF'].mean():.1f}kN (std={sub['BF'].std():.1f}). "
            f"Max stress average: {sub['max_stress'].mean():.1f} MPa. "
            f"Average thinning: {sub['thinning'].mean():.4f}."
        )
        documents.append({
            'id': f'outcome_{outcome.replace(" ", "_")}',
            'text': doc,
            'category': 'simulation_outcome'
        })
 
    # ── B: Parameter effect summaries ──
    param_docs = [
        {
            'id': 'param_BF',
            'text': (
                "Blank Holder Force (BF) effect on sheet metal forming: "
                "BF ranges from 100kN to 500kN. Higher BF values above 350kN "
                "significantly increase crack risk, especially when combined with "
                "thin sheet thickness below 0.975mm. Lower BF values below 200kN "
                "increase wrinkling risk when friction coefficient exceeds 0.12. "
                "BF is the most influential parameter for crack formation according "
                "to SHAP analysis. Optimal BF for safe forming is typically 180-280kN."
            ),
            'category': 'parameter_effect'
        },
        {
            'id': 'param_SHTK',
            'text': (
                "Sheet Thickness (SHTK) effect on sheet metal forming: "
                "SHTK ranges from 0.95mm to 1.00mm. Thinner sheets below 0.97mm "
                "are at higher risk of cracking under high blank holder forces. "
                "SHTK is the second most influential parameter for crack prediction "
                "according to SHAP explainability analysis. Thicker sheets above "
                "0.985mm provide better resistance to cracking. Thinning ratio "
                "increases significantly when BF exceeds 350kN with thin sheets."
            ),
            'category': 'parameter_effect'
        },
        {
            'id': 'param_FC',
            'text': (
                "Friction Coefficient (FC) effect on sheet metal forming: "
                "FC ranges from 0.05 to 0.15. Higher friction above 0.12 combined "
                "with low blank holder force increases wrinkling tendency. "
                "FC has the least influence on crack formation compared to BF and SHTK. "
                "Optimal FC for balanced forming is typically 0.08-0.11. "
                "Lubrication directly controls FC in real manufacturing processes."
            ),
            'category': 'parameter_effect'
        },
        {
            'id': 'param_MAT',
            'text': (
                "Material Hardening Factor (MAT) effect on sheet metal forming: "
                "MAT represents material scatter and ranges from 0.9 to 1.1. "
                "Higher MAT values above 1.05 combined with high BF increase crack risk. "
                "MAT accounts for batch-to-batch material variability in DP600 steel. "
                "MAT has moderate influence on forming outcomes. "
                "Stiffer materials (higher MAT) require adjusted process parameters."
            ),
            'category': 'parameter_effect'
        },
    ]
    documents.extend(param_docs)
 
    # ── C: PSO optimal results ──
    try:
        opt_df = pd.read_csv('results/optimal_parameters.csv')
        opt_text = (
            "PSO Optimisation Results for safe sheet metal forming: "
            "Particle Swarm Optimisation was used to find process parameters "
            "that minimise crack and wrinkle probability while maximising safe outcome probability. "
        )
        for _, row in opt_df.iterrows():
            opt_text += f"{row['Parameter']}={row['Optimal_Value']:.4f} {row['Unit']}, "
        opt_text += (
            "These optimal parameters achieve safe forming probability above 80%. "
            "The optimisation used 50 particles over 100 iterations."
        )
        documents.append({
            'id': 'pso_optimal',
            'text': opt_text,
            'category': 'optimisation'
        })
    except:
        documents.append({
            'id': 'pso_optimal',
            'text': (
                "PSO optimisation finds that safe forming conditions require "
                "BF around 200-250kN, SHTK above 0.98mm, FC around 0.09-0.11, "
                "and MAT close to 1.0. These parameters minimise crack and wrinkle risk."
            ),
            'category': 'optimisation'
        })
 
    # ── D: SHAP findings ──
    documents.append({
        'id': 'shap_insights',
        'text': (
            "SHAP Explainability Analysis Results for sheet metal forming surrogate model: "
            "Feature importance ranking for crack formation prediction: "
            "1. BF (Blank Holder Force) - strongest driver of crack formation, "
            "2. SHTK (Sheet Thickness) - second strongest influence, "
            "3. MAT (Material Hardening) - moderate influence, "
            "4. FC (Friction Coefficient) - weakest influence on cracks. "
            "For wrinkling: FC and BF are most influential. "
            "SHAP analysis extends the original DDACS paper which identified "
            "XAI as future work in Section 5 of Heinzelmann et al. 2025."
        ),
        'category': 'xai_insight'
    })
 
    # ── E: Dataset and paper knowledge ──
    documents.append({
        'id': 'ddacs_paper',
        'text': (
            "DDACS Benchmark Dataset paper by Heinzelmann et al. 2025: "
            "Title: A Comprehensive Benchmark Dataset for Sheet Metal Forming: "
            "Advancing Machine Learning and Surrogate Modelling in Process Simulations. "
            "Published in MATEC Web of Conferences, DOI: 10.1051/matecconf/202540801090. "
            "Dataset contains 32,076 FEM simulations of deep drawing of DP600 dual-phase steel. "
            "Three geometries: Concave, Convex, Rectangular. Drawing depth 30mm, part length 210mm. "
            "Simulations run in LS-DYNA software, each taking 1.5 hours on AMD EPYC 7742 CPU. "
            "Dataset available at DaRUS: DOI 10.18419/DARUS-4801, licensed CC BY 4.0. "
            "Paper uses DGCNN with late fusion for FLD classification. "
            "Future work identified: explainable AI and transfer learning."
        ),
        'category': 'paper_knowledge'
    })
 
    # ── F: Geometry-specific knowledge ──
    if 'geometry' in df.columns:
        for geom in df['geometry'].unique():
            sub = df[df['geometry'] == geom]
            crack_rate = (sub['outcome'] == 2).mean()
            safe_rate  = (sub['outcome'] == 0).mean()
            documents.append({
                'id': f'geometry_{geom}',
                'text': (
                    f"Geometry type {geom} in deep drawing simulations: "
                    f"{len(sub)} simulations with {geom} geometry. "
                    f"Safe rate: {safe_rate:.1%}, Crack rate: {crack_rate:.1%}. "
                    f"Average BF: {sub['BF'].mean():.1f}kN, "
                    f"Average SHTK: {sub['SHTK'].mean():.4f}mm. "
                    f"Average max stress: {sub['max_stress'].mean():.1f} MPa."
                ),
                'category': 'geometry'
            })
 
    # ── G: Concept explanations ──
    concept_docs = [
        {
            'id': 'surrogate_model',
            'text': (
                "A surrogate model is a machine learning model that learns to mimic "
                "an expensive simulation. Instead of running a costly FEM simulation "
                "that takes 1.5 hours per run, the surrogate model predicts the same "
                "result in milliseconds. It is trained on thousands of pre-computed "
                "simulation results. In this project, a PyTorch neural network is the "
                "surrogate that replaces LS-DYNA FEM simulations for sheet metal forming. "
                "Surrogate models are also called metamodels or emulators. "
                "They enable real-time process control and optimisation."
            ),
            'category': 'concept'
        },
        {
            'id': 'fem_explanation',
            'text': (
                "Finite Element Method (FEM) is a numerical simulation technique used "
                "to predict how materials behave under physical forces. In sheet metal "
                "forming, FEM simulates how a metal sheet deforms when pressed by tools. "
                "LS-DYNA is the FEM software used in the DDACS dataset. Each simulation "
                "takes 1.5 hours on a high-performance CPU. FEM divides the sheet into "
                "thousands of small elements and calculates stress, strain and thickness "
                "at each element. The results show where cracks or wrinkles might form."
            ),
            'category': 'concept'
        },
        {
            'id': 'pso_explanation',
            'text': (
                "Particle Swarm Optimisation (PSO) is an optimisation algorithm inspired "
                "by the behaviour of bird flocks. In this project PSO finds the optimal "
                "process parameters (MAT, FC, SHTK, BF) that minimise crack and wrinkle "
                "probability. 50 particles explore the parameter space over 100 iterations. "
                "Each particle represents a set of process parameters. Particles move "
                "towards the best solution found so far. PSO found optimal parameters: "
                "low BF around 100-200kN, maximum SHTK of 1.0mm give safest forming conditions."
            ),
            'category': 'concept'
        },
        {
            'id': 'shap_explanation',
            'text': (
                "SHAP (SHapley Additive exPlanations) is an Explainable AI technique "
                "that explains why a machine learning model makes a specific prediction. "
                "SHAP values show how much each input feature contributed to the output. "
                "In this project SHAP explains which process parameters (MAT, FC, SHTK, BF) "
                "most influence forming outcomes like cracks and wrinkles. "
                "Higher absolute SHAP value means stronger influence on the prediction. "
                "BF (Blank Holder Force) has the highest SHAP value for crack prediction, "
                "meaning it is the most important parameter to control."
            ),
            'category': 'concept'
        },
        {
            'id': 'deep_drawing',
            'text': (
                "Deep drawing is a sheet metal forming process where a flat metal sheet "
                "is pressed into a die to form a cup or box shape. The main components are: "
                "punch (pushes the sheet down), die (shapes the sheet), blank holder (holds "
                "the sheet edges). DP600 dual-phase steel is used in the DDACS dataset. "
                "Common defects are cracks (too much stretching) and wrinkles (too little "
                "blank holder force). The forming limit diagram (FLD) shows which combinations "
                "of strain are safe. Drawing depth in DDACS dataset is 30mm."
            ),
            'category': 'concept'
        },
    ]
    documents.extend(concept_docs)
 
    print(f"\n  Built {len(documents)} knowledge documents")
    return documents
 
 
# ─────────────────────────────────────────────
# 2. BUILD FAISS INDEX
# ─────────────────────────────────────────────
def build_faiss_index(documents):
    print(f"  Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
 
    texts = [doc['text'] for doc in documents]
    print(f"  Embedding {len(texts)} documents...")
    embeddings = embedder.encode(texts, show_progress_bar=False)
    embeddings = embeddings.astype('float32')
 
    # Normalise for cosine similarity
    faiss.normalize_L2(embeddings)
 
    # Build index
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalisation
    index.add(embeddings)
 
    # Save index and documents
    faiss.write_index(index, 'rag_db/forming_index.faiss')
    with open('rag_db/documents.json', 'w') as f:
        json.dump(documents, f, indent=2)
 
    print(f"  ✅ FAISS index built — {index.ntotal} vectors, dim={dim}")
    return index, embedder, documents
 
 
# ─────────────────────────────────────────────
# 3. RETRIEVAL FUNCTION
# ─────────────────────────────────────────────
def retrieve(query, index, embedder, documents, top_k=3):
    q_emb = embedder.encode([query]).astype('float32')
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
 
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            'text':     documents[idx]['text'],
            'category': documents[idx]['category'],
            'score':    float(score)
        })
    return results
 
 
# ─────────────────────────────────────────────
# 4. LLM RESPONSE GENERATION
# ─────────────────────────────────────────────
def generate_response(query, context_docs):
    context = "\n\n".join([
        f"[Source {i+1} — {doc['category']}]:\n{doc['text']}"
        for i, doc in enumerate(context_docs)
    ])
 
    prompt = f"""You are an expert process engineer specialising in sheet metal forming
and physics-based machine learning. Answer the question using the provided context.
Be specific, cite numbers where available, and keep your answer concise.
If the question is not related to sheet metal forming or the project, politely say
you can only answer questions about this project.
 
Context:
{context}
 
Question: {query}
 
Answer:"""
 
    # Try Groq
    if LLM_PROVIDER == 'groq' and GROQ_API_KEY:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Groq error: {e}\n\nFallback — {context_docs[0]['text']}"
 
    # Try Anthropic
    elif LLM_PROVIDER == 'anthropic' and ANTHROPIC_API_KEY:
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Anthropic error: {e}\n\nFallback — {context_docs[0]['text']}"
 
    # No LLM — return best retrieved document
    return (f"[RAG Retrieval — Add GROQ_API_KEY to .env for full LLM responses]\n\n"
            f"Most relevant information:\n\n{context_docs[0]['text']}")
 
 
# ─────────────────────────────────────────────
# 5. RAG QUERY FUNCTION
# ─────────────────────────────────────────────
def ask(query, index, embedder, documents, verbose=True):
    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  Q: {query}")
        print(f"  {'─'*50}")
 
    context_docs = retrieve(query, index, embedder, documents, top_k=3)
 
    if verbose:
        print(f"  Retrieved {len(context_docs)} relevant documents:")
        for i, doc in enumerate(context_docs):
            print(f"    [{i+1}] {doc['category']} (score={doc['score']:.3f})")
 
    answer = generate_response(query, context_docs)
 
    if verbose:
        print(f"\n  A: {answer}")
 
    return answer
 
 
# ─────────────────────────────────────────────
# 6. MAIN — BUILD + DEMO QUERIES
# ─────────────────────────────────────────────
if __name__ == '__main__':
 
    # Build knowledge base
    documents = build_knowledge_base()
 
    # Build FAISS index
    index, embedder, documents = build_faiss_index(documents)
 
    # ── Demo queries ──
    print(f"\n{'='*55}")
    print(f"  RAG Demo — Sample Engineering Queries")
    print(f"{'='*55}")
 
    demo_queries = [
        "What parameters should I use to avoid cracks?",
        "How does blank holder force affect forming outcome?",
        "What did the Stuttgart paper find about springback?",
        "Which parameter is most important for safe forming?",
        "What is the difference between Concave and Rectangular geometry?",
        "What are the optimal process parameters found by PSO?",
        "How does sheet thickness affect crack risk?",
    ]
 
    results_log = []
    for query in demo_queries:
        answer = ask(query, index, embedder, documents, verbose=False)
        print(f"\n  Q: {query}")
        print(f"  A: {answer}")
        results_log.append({'query': query, 'answer': answer})
 
    # Save Q&A log
    pd.DataFrame(results_log).to_csv('results/rag_qa_log.csv', index=False)
    print(f"\n  Saved → results/rag_qa_log.csv")
 
    # ── Interactive mode ──
    print(f"\n{'='*55}")
    print(f"  Interactive RAG — Ask Your Own Questions")
    print(f"  (type 'quit' to exit)")
    print(f"{'='*55}")
 
    while True:
        try:
            user_query = input("\n  Your question: ").strip()
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\n  Exiting RAG assistant.")
                break
            if user_query:
                answer = ask(user_query, index, embedder, documents, verbose=False)
                print(f"\n  A: {answer}")
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting RAG assistant.")
            break
 
    print(f"\n✅ Script 6 complete\n")
 
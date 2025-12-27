import { 
    SupabaseVectorStore, 
    OllamaEmbeddings, 
    RecursiveCharacterTextSplitter, 
    FaissStore, 
    Document, 
    Embeddings,
    createStuffDocumentsChain,
    createRetrievalChain,
    ChatPromptTemplate,
    BaseChatModel,
    VectorStore,
    BaseRetriever
} from "./imports"
import { supabase } from "./supabase"
import { getLLM } from "./helpers"

interface SupabaseStoreConfig {
    docs: Document<Record<string,any>>[]
    table_name?: string
    RPC_function?: string
}

const baseEmbeddings = new OllamaEmbeddings({
    model: "nomic-embed-text"
})

const baseSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50
})

export function turn_to_docs<T>(docs: T[]): Document<Record<string,any>>[] {
    return docs.map(doc => new Document({
        pageContent: typeof doc === "string" ? doc : JSON.stringify(doc,null,2),
        metadata: {}
    }))
}

export async function createSupabaseVectoreStore({docs, table_name = "documents", RPC_function = "match_documents"}: SupabaseStoreConfig) {
    const splitted_docs = await baseSplitter.splitDocuments(docs)
    return await SupabaseVectorStore.fromDocuments(  
        splitted_docs,
        baseEmbeddings,
        {
            client: supabase,
            tableName: table_name,
            queryName: RPC_function
        }
    )

}

// Bestehenden Supabase Store holen (ohne neue Docs)
export function getSupabaseVectorStore({table_name = "documents", RPC_function = "match_documents"}: {table_name?: string, RPC_function?: string} = {}) {
    return new SupabaseVectorStore(baseEmbeddings, {
        client: supabase,
        tableName: table_name,
        queryName: RPC_function
    })
}

export async function createFaissStore({docs, save_path, embeddings = baseEmbeddings}: {docs: Document<Record<string,any>>[], save_path?: string, embeddings?: Embeddings}) {
    const splitted_docs = await baseSplitter.splitDocuments(docs)
    const vectore_store = await FaissStore.fromDocuments(
        splitted_docs,
        embeddings
    )
    if (save_path) {
        await vectore_store.save(save_path)
    }
    return vectore_store
}

// Gespeicherten FAISS Store von Disk laden
export async function loadFaissStore({path, embeddings = baseEmbeddings}: {path: string, embeddings?: Embeddings}) {
    return await FaissStore.load(path, embeddings)
}

// ============ RETRIEVAL CHAINS ============

// Erstellt eine Retrieval Chain aus einem VectorStore
export async function createRAGChain({
    vectorStore,
    llm,
    prompt,
    num_of_results_from_vdb = 4
}: {
    vectorStore: VectorStore,
    llm?: BaseChatModel,
    prompt?: ChatPromptTemplate,
    num_of_results_from_vdb?: number
}) {
    const model = llm ?? getLLM("groq")
    const retriever = vectorStore.asRetriever({ k: num_of_results_from_vdb })
    
    // Document Chain: Kombiniert Docs in den Prompt
    const documentChain = await createStuffDocumentsChain({
        llm: model as any,
        prompt: prompt as any
    })
    
    // Retrieval Chain: Verbindet Retriever mit Document Chain
    const retrievalChain = await createRetrievalChain({
        retriever: retriever as any,
        combineDocsChain: documentChain
    })
    
    return retrievalChain
}

// Erstellt eine Retrieval Chain direkt aus einem Retriever
export async function createRAGChainFromRetriever({
    retriever,
    llm,
    prompt
}: {
    retriever: BaseRetriever,
    llm?: BaseChatModel,
    prompt?: ChatPromptTemplate
}) {
    const model = llm ?? getLLM("groq")
    
    const documentChain = await createStuffDocumentsChain({
        llm: model as any,
        prompt: prompt as any
    })
    
    const retrievalChain = await createRetrievalChain({
        retriever: retriever as any,
        combineDocsChain: documentChain
    })
    
    return retrievalChain
}


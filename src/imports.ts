import { BaseMessage, HumanMessage, AIMessage, SystemMessage} from '@langchain/core/messages'
import { ChatPromptTemplate,MessagesPlaceholder } from '@langchain/core/prompts'
import { BaseOutputParser,StructuredOutputParser,StringOutputParser } from '@langchain/core/output_parsers'
import { BaseChatModel } from '@langchain/core/language_models/chat_models'
import { Document } from '@langchain/core/documents'
import { BaseRetriever } from '@langchain/core/retrievers'
import { Runnable, type RunnableConfig } from '@langchain/core/runnables'
import { DynamicStructuredTool, tool } from '@langchain/core/tools'
import { Embeddings } from '@langchain/core/embeddings'

// LLM Providers
import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama'
import { ChatGroq } from '@langchain/groq'
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai'
import { ChatAnthropic } from '@langchain/anthropic'

// RAG: Vector Stores (Base Classes)
import { VectorStore, SaveableVectorStore, VectorStoreRetriever } from '@langchain/core/vectorstores'

// RAG: Vector Store Implementations
import { SupabaseVectorStore } from '@langchain/community/vectorstores/supabase'
import { FaissStore } from '@langchain/community/vectorstores/faiss'
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory'

// RAG: Chains
import { createStuffDocumentsChain } from '@langchain/classic/chains/combine_documents'
import { createRetrievalChain } from '@langchain/classic/chains/retrieval'
import { BaseChain } from '@langchain/classic/chains'

// RAG: Text Splitting
import { RecursiveCharacterTextSplitter, CharacterTextSplitter } from '@langchain/textsplitters'

// Agents
import { createReactAgent, ToolNode } from '@langchain/langgraph/prebuilt'
import { MemorySaver, BaseCheckpointSaver } from '@langchain/langgraph'
import { PostgresSaver } from '@langchain/langgraph-checkpoint-postgres'
import { SqliteSaver } from '@langchain/langgraph-checkpoint-sqlite'
import { type Checkpoint, type CheckpointMetadata, type LangGraphRunnableConfig } from "@langchain/langgraph"
import type { CheckpointTuple, CheckpointListOptions, PendingWrite, ChannelVersions } from "@langchain/langgraph-checkpoint"

import { z } from 'zod'
import dotenv from 'dotenv'
dotenv.config()

// Core
export {
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ChatPromptTemplate,
    MessagesPlaceholder,
    BaseOutputParser,
    StructuredOutputParser,
    StringOutputParser,
    BaseChatModel,
    Document,
    BaseRetriever,
    Runnable,
    DynamicStructuredTool,
    tool,
}

// LLM Providers
export {
    ChatOllama,
    ChatGroq,
    ChatOpenAI,
    ChatAnthropic,
}

// Embeddings
export {
    Embeddings,        // Base class für alle Embeddings
    OllamaEmbeddings,
    OpenAIEmbeddings,
}

// RAG: Vector Stores
export {
    VectorStore,           // Base class für alle Vector Stores
    SaveableVectorStore,   // Base class für speicherbare Vector Stores
    VectorStoreRetriever,  // Retriever für Vector Stores
    SupabaseVectorStore,   // Cloud: Supabase
    FaissStore,            // Lokal: Speichert in Datei
    MemoryVectorStore,     // Lokal: In-Memory (kein Speichern)
}

// RAG: Chains & Utils
export {
    createStuffDocumentsChain,
    createRetrievalChain,
    BaseChain,
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
}

// Agents
export {
    createReactAgent,
    ToolNode,
    MemorySaver,
    BaseCheckpointSaver,
}

// Checkpoint Savers (optional - Packages müssen installiert sein)
export {
    PostgresSaver,
    SqliteSaver,
}

// Checkpoint
export {
    type Checkpoint,
    type CheckpointMetadata,
    type LangGraphRunnableConfig,
    type RunnableConfig,
    type CheckpointTuple,
    type CheckpointListOptions,
    type PendingWrite,
    type ChannelVersions,
}

// Utils
export {
    z,
    dotenv
}
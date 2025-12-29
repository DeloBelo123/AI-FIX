import { z } from "zod/v3"
import { DynamicStructuredTool, Runnable } from "../imports"
import { BaseChatModel } from "../imports"
import { BaseCheckpointSaver } from "../imports"
import { VectorStore } from "../imports"
import { turn_to_docs } from "../rag"
import { getLLM, wait } from "../helpers"
import { createReactAgent } from "../imports"
import { HumanMessage, AIMessage, MemorySaver } from "../imports"
import { structure } from "../helpers"
import { SmartCheckpointSaver } from "../memory"

interface AgentProps<T extends z.ZodObject<any,any>>{
    prompt?: Array<["system", string]>
    tools: DynamicStructuredTool[]
    llm?: BaseChatModel
    schema?: T
    memory?: BaseCheckpointSaver
}

/**
 * CONSTRUCTOR:
 * @example constructor({
        prompt = [["system", `Du bist ein hilfreicher Assistent. 
            WICHTIG: 
            - Nutze Tools NUR wenn nötig
            - Nach jedem Tool-Call gib eine finale Antwort
            - Stoppe nach der Antwort, rufe keine Tools mehr auf
            - Wenn du die Antwort hast, gib sie direkt zurück`]],
        tools,
        llm = getLLM("groq"),
        schema,
        memory = new SmartCheckpointSaver(new MemorySaver())
    }: AgentProps<T>) {
        this.prompt = prompt
        this.tools = tools
        this.llm = llm
        this.schema = schema
        this.memory = memory
    }
 */
export class Agent<T extends z.ZodObject<any,any>> {
    private prompt: Array<["system", string]>
    private tools: DynamicStructuredTool[]
    private llm: BaseChatModel
    private schema: T | undefined
    private agent: any
    private memory: BaseCheckpointSaver | undefined
    private vectorStore: VectorStore | undefined
    private rag_tool: DynamicStructuredTool | undefined
    private times_of_added_context: number = 0

    constructor({
        prompt = [["system", `Du bist ein hilfreicher Assistent. 
            WICHTIG: 
            - Nutze Tools NUR wenn nötig
            - Nach jedem Tool-Call gib eine finale Antwort
            - Stoppe nach der Antwort, rufe keine Tools mehr auf
            - Wenn du die Antwort hast, gib sie direkt zurück`]],
        tools,
        llm = getLLM("groq"),
        schema,
        memory = new SmartCheckpointSaver(new MemorySaver())
    }: AgentProps<T>) {
        this.prompt = prompt
        this.tools = tools
        this.llm = llm
        this.schema = schema
        this.memory = memory
    }

    public async invoke(invokeInput: Record<string, any> & { input: string, thread_id?: string, debug?: boolean }): Promise<T extends undefined ? string : z.infer<T>> {
        const { input, thread_id, debug, ...variables } = invokeInput
        
        // Dynamische Variablen als System-Messages (thread_id wird NICHT als Message, sondern als config genutzt)
        const contextMessages: Array<["system", string]> = Object.entries(variables).map(
            ([key, value]) => ["system", `${key}: ${typeof value === "object" ? JSON.stringify(value) : value}`]
        )

        // Tools für diesen invoke (inkl. RAG falls vorhanden)
        const activeTools = this.rag_tool 
            ? [...this.tools, this.rag_tool] 
            : this.tools

        this.agent = createReactAgent({
            llm: this.llm as any,
            tools: activeTools as any,
            checkpointSaver: this.memory as any,
            prompt: (state) => [
                ...this.prompt,
                ...contextMessages,
                ...state.messages
            ] 
        })
        
        const config = thread_id && this.memory ? { configurable: { thread_id } } : undefined
        const result = await this.agent.invoke({ messages: [new HumanMessage(input)] } as any, config)
        if(debug) return result
        const lastMessage = result.messages[result.messages.length - 1]
        const content = lastMessage.content

        if (this.schema) {
            return await structure({ data: content, into: this.schema, llm: this.llm }) as any
        }
        return content as any
    }

    public async *stream(invokeInput: Record<string, any> & { input: string, thread_id?: string, debug?: boolean }): AsyncGenerator<string, void, unknown> {
        const { input, thread_id, debug, ...variables } = invokeInput
        
        const contextMessages: Array<["system", string]> = Object.entries(variables).map(
            ([key, value]) => ["system", `${key}: ${typeof value === "object" ? JSON.stringify(value) : value}`]
        )

        const activeTools = this.rag_tool 
            ? [...this.tools, this.rag_tool] 
            : this.tools

        this.agent = createReactAgent({
            llm: this.llm as any,
            tools: activeTools as any,
            checkpointSaver: this.memory as any,
            prompt: (state) => [
                ...this.prompt,
                ...contextMessages,
                ...state.messages
            ] 
        })
        
        const config = thread_id && this.memory ? { configurable: { thread_id } } : undefined
        
        const response = await this.agent.invoke({ messages: [new HumanMessage(input)] }, config)
        
        // Finde die letzte AIMessage (nicht einfach die letzte Message, da es auch ToolMessages geben kann)
        const messages = response.messages
        let lastAIMessage = null
        
        // Durchlaufe rückwärts, um die letzte AIMessage zu finden
        for (let i = messages.length - 1; i >= 0; i--) {
            const msg = messages[i]
            // Prüfe ob es eine AIMessage ist (verschiedene Möglichkeiten)
            if (msg instanceof AIMessage || 
                msg._getType?.() === 'ai' || 
                msg.constructor?.name === 'AIMessage' ||
                (msg.id && msg.id.includes('AIMessage'))) {
                lastAIMessage = msg
                break
            }
        }
        
        if (!lastAIMessage) {
            // Fallback: Wenn keine AIMessage gefunden, nimm die letzte Message
            lastAIMessage = messages[messages.length - 1]
        }
        
        // Extrahiere Content (kann String oder Array sein)
        let content = ''
        if (typeof lastAIMessage.content === 'string') {
            content = lastAIMessage.content
        } else if (Array.isArray(lastAIMessage.content)) {
            // Content kann ein Array von Content-Blöcken sein
            content = lastAIMessage.content
                .filter((block: any) => block.type === 'text' || typeof block === 'string')
                .map((block: any) => typeof block === 'string' ? block : block.text || '')
                .join(' ')
        } else {
            content = String(lastAIMessage.content || '')
        }
        
        // Wenn Content leer ist, gib nichts zurück
        if (!content || content.trim().length === 0) {
            return
        }
        
        // Teile den Content in Wörter auf
        const words = content.split(" ")
        
        // Yield jedes Wort
        for (const word of words) {
            yield word + " "
        }
    }

    public setContext(vectorStore: VectorStore, metadata: { name?: string, description?: string } = {}) {
        this.vectorStore = vectorStore
        this.rag_tool = new DynamicStructuredTool({
            name: metadata.name ?? "search_context",
            description: metadata.description ?? "Search the knowledge base for relevant information",
            schema: z.object({
                query: z.string().describe("The search query")
            }),
            func: async ({ query }) => {
                const docs = await this.vectorStore?.similaritySearch(query, 3)
                if (!docs || docs.length === 0) return "No relevant information found."
                return docs.map(doc => doc.pageContent).join("\n\n---\n\n")
            }
        })
    }

    public async addContext(data: Array<any>){
        if(!this.vectorStore) {
            throw new Error("Cant add context, no vector store set")
        }
        this.times_of_added_context++
        const docs = turn_to_docs(data)
        await this.vectorStore.addDocuments(docs)
        console.log(`Added context ${this.times_of_added_context} ${this.times_of_added_context === 1 ? "time" : "times"}`)
    }

    public clearContext(){
        this.rag_tool = undefined
        this.vectorStore = undefined
        this.times_of_added_context = 0
        console.log("Context cleared")
    }

    public get currentTools(): string[] {
        const tools = [...this.tools]
        if (this.rag_tool) {
            tools.push(this.rag_tool)
        }
        return tools.map(tool => tool.name)
    }
    
    public hasContext(): boolean {
        return this.vectorStore !== undefined && this.rag_tool !== undefined
    }
}
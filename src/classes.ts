import { 
    BaseChatModel,
    StructuredOutputParser,
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    VectorStore,
    createRetrievalChain, 
    createStuffDocumentsChain,
    createReactAgent,
    BaseCheckpointSaver,
    DynamicStructuredTool,
    HumanMessage,
    Runnable
} from "./imports"
import { turn_to_docs } from "./rag"
import { structure } from "./helpers"
import { getLLM } from "./helpers"
import { z } from "zod/v3"

interface ChainProps<T extends z.ZodObject<any,any>>{
    prompt?:Array<["human" | "system", string] | MessagesPlaceholder<any>>
    llm?:BaseChatModel
    schema?:T
}

export const DEFAULT_SCHEMA = z.object({ 
    output: z.string().describe("Die Antwort auf die Frage") 
})

export class Chain<T extends z.ZodObject<any,any> = typeof DEFAULT_SCHEMA> {
    private prompt:Array<["human" | "system", string] | MessagesPlaceholder<any>>
    private vectorStore: VectorStore | undefined
    private times_of_added_context: number = 0
    private parser:StructuredOutputParser<T>
    private llm:BaseChatModel
    private schema:T

    constructor({prompt = [["system","du bist ein hilfreicher Assistent"]],llm = getLLM("groq"),schema}:ChainProps<T> = {}){
        this.prompt = prompt
        this.llm = llm
        this.schema = (schema ?? DEFAULT_SCHEMA) as unknown as T
        this.parser = StructuredOutputParser.fromZodSchema(this.schema)
    }

    public async invoke(input:Record<string,any>):Promise<z.infer<T>>{
        const messagesArray = [...this.prompt]
        messagesArray.push(["system", "You MUST respond ONLY with valid JSON matching this exact schema:\n{format_instructions}\n\nIMPORTANT: \n- Output ONLY valid JSON, no markdown code blocks\n- No backslashes or line breaks in strings\n- All strings must be on single lines\n- Do NOT wrap in ```json``` blocks\n- Return the JSON object DIRECTLY"])
        if(this.vectorStore) messagesArray.push(["system", "Hier ist relevanter Kontext:\n{context}"])
        for(const key in input){
            messagesArray.push(["human",`{${key}}`])
        }
        const true_prompt = ChatPromptTemplate.fromMessages(messagesArray)
        const invokeInput = {...input, format_instructions: this.parser.getFormatInstructions()}
        
        if(this.vectorStore){
            const retriever = this.vectorStore.asRetriever()
            const stuff_chain = await createStuffDocumentsChain({
                llm: this.llm as any,
                prompt: true_prompt as any,
                outputParser: this.parser as any
            })
            const chain = await createRetrievalChain({
                combineDocsChain: stuff_chain,
                retriever: retriever as any
            })
            console.log("created retrieval chain")
            const respo = await chain.invoke({input: JSON.stringify(input), ...invokeInput})
            return this.schema.parse(respo.answer)
        }
        
        const chain = true_prompt.pipe(this.llm as any).pipe(this.parser)
        console.log("created normal chain")
        const respo = await chain.invoke(invokeInput)
        return this.schema.parse(respo) 
    }

    /** Fügt RAG-Kontext hinzu. Docs werden EINMAL zum VectorStore hinzugefügt. */
    public async addContext(data: Array<any>){
        if(!this.vectorStore) {
            throw new Error("Cant add context, no vector store set")
        }
        this.times_of_added_context++
        const docs = turn_to_docs(data)
        await this.vectorStore.addDocuments(docs)
        console.log(`Added context ${this.times_of_added_context} ${this.times_of_added_context === 1 ? "time" : "times"}`)
    }

    public async setContext(vectorStore: VectorStore){
        console.log("Setting context")
        this.vectorStore = vectorStore
    }

    public clearContext(){
        this.vectorStore = undefined
        this.times_of_added_context = 0
        console.log("Context cleared")
    }
}


interface AgentProps<T extends z.ZodObject<any,any>>{
    prompt?: Array<["system", string]>
    tools: DynamicStructuredTool[]
    llm?: BaseChatModel
    schema?: T
    memory?: BaseCheckpointSaver
}

export class Agent<T extends z.ZodObject<any,any> = typeof DEFAULT_SCHEMA> {
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
        prompt = [["system","Du bist ein hilfreicher Assistent"]],
        tools,
        llm = getLLM("groq"),
        schema,
        memory
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
            ] as any
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

    public getTools(): string[] {
        return this.tools.map(tool => tool.name)
    }
}

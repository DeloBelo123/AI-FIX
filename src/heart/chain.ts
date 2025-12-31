import { 
    BaseChatModel,
    StructuredOutputParser,
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    VectorStore,
    createRetrievalChain, 
    createStuffDocumentsChain,
} from "../imports"
import { turn_to_docs } from "../rag"
import { getLLM } from "../helpers"
import { z } from "zod/v3"

interface ChainProps<T extends z.ZodObject<any,any>>{
    prompt?:Array<["human" | "system", string] | MessagesPlaceholder<any>>
    llm?:BaseChatModel
    schema?:T
}

export const DEFAULT_SCHEMA = z.object({ 
    output: z.string().describe("Dein Output zur anfrage des Users") 
})

/**
 * CONSTRUCTOR:
 * @example constructor({prompt = [["system","du bist ein hilfreicher Assistent"]],llm = getLLM("groq"),schema}:ChainProps<T> = {}){
        this.prompt = prompt
        this.llm = llm
        this.schema = (schema ?? DEFAULT_SCHEMA) as unknown as T
        this.parser = StructuredOutputParser.fromZodSchema(this.schema)
    }
 */
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

    public async invoke(input:Record<string,any> & {debug?: boolean}):Promise<z.infer<T>>{
        const messagesArray = [...this.prompt]
        messagesArray.push(["system", "You MUST respond ONLY with valid JSON matching this exact schema:\n{format_instructions}\n\nIMPORTANT: \n- Output ONLY valid JSON, no markdown code blocks\n- No backslashes or line breaks in strings\n- All strings must be on single lines\n- Do NOT wrap in ```json``` blocks\n- Return the JSON object DIRECTLY"])
        if(this.vectorStore) messagesArray.push(["system", "Hier ist relevanter Kontext:\n{context}"])
        for(const key in input){
            if(key === "debug") continue
            if(key === "thread_id"){
                console.error("eine normale chain hat keine memory, deswegen wird thread_id ignoriert")
                continue
            } 
            if(typeof input[key] !== "string"){
                messagesArray.push(["human",`${key}:${JSON.stringify(input[key])}`])
            } else {
                messagesArray.push(["human",`${key}:{${key}}`])
            }
        }
        const true_prompt = ChatPromptTemplate.fromMessages(messagesArray)
        if(input.debug) console.log("Prompt: ", true_prompt)
        const invokeInput = {...input, format_instructions: this.parser.getFormatInstructions()}
        
        if(this.vectorStore){
            const retriever = this.vectorStore.asRetriever()
            const stuff_chain = await createStuffDocumentsChain({
                llm: this.llm,
                prompt: true_prompt,
                outputParser: this.parser
            })
            const chain = await createRetrievalChain({
                combineDocsChain: stuff_chain,
                retriever: retriever
            })
            if (input.debug) console.log("created retrieval chain")
            const respo = await chain.invoke({input: JSON.stringify(input), ...invokeInput})
            return this.schema.parse(respo.answer)
        }
        
        const chain = true_prompt.pipe(this.llm).pipe(this.parser)
        if (input.debug) console.log("created normal chain")
        const respo = await chain.invoke(invokeInput)
        return this.schema.parse(respo) 
    }


    public async *stream(input: Record<string, any>): AsyncGenerator<string, void, unknown> {
        const messagesArray = [...this.prompt]
        // Beim Streamen KEIN Schema-Prompt - nur reiner Text
        if(this.vectorStore) messagesArray.push(["system", "Hier ist relevanter Kontext:\n{context}"])
            for(const key in input){
                if(key === "debug") continue
                if(key === "thread_id"){
                    console.error("eine normale chain hat keine memory, deswegen wird thread_id ignoriert")
                    continue
                } 
                if(typeof input[key] !== "string"){
                    messagesArray.push(["human",`${key}:${JSON.stringify(input[key])}`])
                } else {
                    messagesArray.push(["human",`${key}:{${key}}`])
                }
            }
        const true_prompt = ChatPromptTemplate.fromMessages(messagesArray)
        if(input.debug) console.log("Prompt: ", true_prompt)
        const invokeInput = {...input}
        
        if(this.vectorStore){
            const retriever = this.vectorStore.asRetriever()
            if(input.debug) console.log("created retrieval chain (streaming)")
            
            // Für RAG: Hole Context und stream dann die LLM-Antwort
            const contextDocs = await retriever.invoke(JSON.stringify(input))
            const contextText = contextDocs.map((doc: any) => doc.pageContent).join("\n\n")
            
            // Stream die LLM-Antwort mit Context
            const streamChain = true_prompt.pipe(this.llm)
            const stream = await streamChain.stream({...invokeInput, context: contextText})
            
            for await (const chunk of stream) {
                if (chunk && typeof chunk === 'object' && 'content' in chunk) {
                    yield chunk.content as string
                } else if (typeof chunk === 'string') {
                    yield chunk
                }
            }
        } else {
            // Ohne RAG: Stream direkt
            const streamChain = true_prompt.pipe(this.llm)
            if(input.debug) console.log("created normal chain (streaming)")
            
            const stream = await streamChain.stream(invokeInput)
            for await (const chunk of stream) {
                if (chunk && typeof chunk === 'object' && 'content' in chunk) {
                    yield chunk.content as string
                } else if (typeof chunk === 'string') {
                    yield chunk
                }
            }
        }
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

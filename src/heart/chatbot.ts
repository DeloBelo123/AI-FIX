import { BaseCheckpointSaver, HumanMessage, AIMessage, LangGraphRunnableConfig, BaseMessage, CheckpointMetadata, VectorStore, DynamicStructuredTool } from "../imports"
import { BaseChatModel } from "../imports"
import { SmartCheckpointSaver } from "../memory"
import { Chain } from "./chain"
import { Agent } from "./agent"
import { MemorySaver } from "../imports"
import { getLLM, logChunk } from "../helpers"
import { input } from "../cli"

type ChatBotProps = { memory?: BaseCheckpointSaver } & ({
    chain: Chain
} | {
    llm?:BaseChatModel
    prompt?: Array<["system", string]>
    tools?: DynamicStructuredTool[]
})

/**
 * CONSTRUCTOR: (Viele Defaults, kannst selber bestimmte dinge überschreiben)
 * @example constructor({memory, ...rest}: ChatBotProps = {}){
        //memory ist immer da
        this.memory = memory ?? new SmartCheckpointSaver(new MemorySaver())

        // SELTEN
        // jetzt gucken wir: soll eine chain einfach zum chatbot gemacht werden? dann mache das zu this.chain
        if ("chain" in rest){
            this.chain = rest.chain

        // hier checken wir jetzt die config-optionen
        // wenn wir tools mitgegeben haben beim config, dann erstellen wir einen agent
        } else if ("tools" in rest){
            this.tools = rest.tools
            this.agent = new Agent({
                tools: this.tools!,
                memory: this.memory,
                llm: rest.llm ?? getLLM("groq"),
                prompt: rest.prompt ?? [["system", "Du bist ein hilfreicher Chatbot der mit dem User ein höffliches und hilfreiches Gespräch führt"]]
            })
        // wenn wir keine tools mitgegeben haben, dann erstellen wir einfach eine chain
        } else {
            this.chain = new Chain({
                llm: rest.llm ?? getLLM("groq"),
                prompt: rest.prompt ?? [["system", "Du bist ein hilfreicher Chatbot der mit dem User ein höffliches und hilfreiches Gespräch führt"]]
            })
        }
    }
 * 
 * @example entweder gibst du direkt eine chain mit memory:
 * @param props.memory 
 * @param props.chain 
 * 
 * @example oder eine chain/agent wird anhand deines llms,tools oder prompts erstellt mit memory:
 * @param props.memory
 * @param props.llm 
 * @param props.prompt 
 */
export class ChatBot {
    private memory: BaseCheckpointSaver
    private chain: Chain | undefined
    private agent:Agent<any> | undefined

    constructor({memory, ...rest}: ChatBotProps = {}){
        //memory ist immer da
        this.memory = memory ?? new SmartCheckpointSaver(new MemorySaver())

        // SELTEN
        // jetzt gucken wir: soll eine chain einfach zum chatbot gemacht werden? dann mache das zu this.chain
        if ("chain" in rest){
            this.chain = rest.chain

        // hier checken wir jetzt die config-optionen
        // wenn wir tools mitgegeben haben beim config, dann erstellen wir einen agent
        } else if ("tools" in rest && rest.tools){
            this.agent = new Agent({
                tools: rest.tools,
                memory: this.memory,
                llm: rest.llm ?? getLLM("groq"),
                prompt: rest.prompt ?? [["system", "Du bist ein hilfreicher Chatbot der mit dem User ein höffliches und hilfreiches Gespräch führt"]]
            })
            
        // wenn wir keine tools mitgegeben haben, dann erstellen wir einfach eine chain
        } else {
            this.chain = new Chain({
                llm: rest.llm ?? getLLM("groq"),
                prompt: rest.prompt ?? [["system", "Du bist ein hilfreicher Chatbot der mit dem User ein höffliches und hilfreiches Gespräch führt"]]
            })
        }
    }

    public async *chat({message, thread_id}: {message: string, thread_id: string}): AsyncGenerator<string, string, unknown> {
        const config: LangGraphRunnableConfig = {
            configurable: { thread_id }
        }
        if (this.agent){
            const chunks: string[] = []
            for await (const chunk of this.agent.stream({
                input: message,
                thread_id: thread_id
            })){
                chunks.push(chunk)
                yield chunk
            }
            return chunks.join('')
        } else if (this.chain){
            const checkpoint = await this.memory.get(config)
        
            let historyMessages: BaseMessage[] = []
            if (checkpoint && checkpoint.channel_values && checkpoint.channel_values.messages) {
                historyMessages = checkpoint.channel_values.messages as BaseMessage[]
            }

            const userMessage = new HumanMessage(message)
            const allMessages = [...historyMessages, userMessage]

            const historyText = allMessages.map(msg => {
                if (msg instanceof HumanMessage) {
                    return `User: ${typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)}`
                } else if (msg instanceof AIMessage) {
                    return `Assistant: ${typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)}`
                } else {
                    return `System: ${typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)}`
                }
            }).join('\n\n')

            const chunks: string[] = []
            
            try {
                for await (const chunk of this.chain.stream({ 
                    message: `${historyText}\n\nUser: ${message}` 
                })) {
                    chunks.push(chunk)
                    yield chunk
                }
            } finally {
                // Memory-Speicherung wird IMMER ausgeführt, auch wenn der Stream fehlschlägt
                const responseText = chunks.join('')

                if (responseText.length > 0) {
                    const aiMessage = new AIMessage(responseText)
                    const newMessages = [...allMessages, aiMessage]
                    
                    const newCheckpoint = {
                        ...checkpoint,
                        channel_values: { 
                            ...(checkpoint?.channel_values || {}),
                            messages: newMessages 
                        },
                        channel_versions: {
                            ...(checkpoint?.channel_versions || {}),
                            messages: newMessages.length
                        },
                        versions_seen: checkpoint?.versions_seen || {},
                        v: (checkpoint?.v || 0) + 1,
                        id: checkpoint?.id || `${thread_id}-${Date.now()}`,
                        ts: checkpoint?.ts || new Date().toISOString()
                    }

                    const metadata: CheckpointMetadata = {
                        source: "input",
                        step: (checkpoint?.v || 0) + 1,
                        parents: {}
                    }

                    await this.memory.put(config, newCheckpoint, metadata, { messages: newMessages.length })
                }
            }
            
            return chunks.join('')
        } else {
            throw new Error("Neither chain nor agent is configured")
        }
    }

    public async session({
        breakword = "exit",
        numberOfMessages = Number.POSITIVE_INFINITY,
        id = `${Date.now()}`
    }:{
        breakword?:string,
        numberOfMessages?:number,
        id?:string
    } = {}){
        let messages = 0
        while(true){
            try{
                const message = await input("You: ")
                if(message === breakword){
                    break
                }
                const response = this.chat({
                    message: message, 
                    thread_id: id,
                })
                console.log("Assistant: ")
                for await (const chunk of response) {
                    logChunk(chunk)
                }
            } catch(e){
                console.error("Error: ", e)
            }
            messages = messages + 2
            if(messages > numberOfMessages){
                break
            }
        }
    }

    public async addContext(data: Array<any>){
        if (this.chain){
            this.chain.addContext(data)
        } else if (this.agent){
            this.agent.addContext(data)
        }
    }

    public async setContext(vectorStore: VectorStore){
        if (this.chain){
            this.chain.setContext(vectorStore)
        } else if (this.agent){
            this.agent.setContext(vectorStore)
        }
    }

    public async clearContext(){
        if (this.chain){
            this.chain.clearContext()
        } else if (this.agent){
            this.agent.clearContext()
        }
    }

}

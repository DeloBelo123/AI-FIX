import { BaseCheckpointSaver, HumanMessage, AIMessage, LangGraphRunnableConfig, BaseMessage, CheckpointMetadata, VectorStore } from "../imports"
import { BaseChatModel } from "../imports"
import { SmartCheckpointSaver } from "../memory"
import { Chain } from "./chain"
import { MemorySaver } from "../imports"
import { getLLM, logChunk } from "../helpers"
import { input } from "../cli"

type ChatBotProps = { memory?: BaseCheckpointSaver } & ({
    chain: Chain
} | {
    llm?:BaseChatModel
    prompt?: Array<["system", string]>
})

/**
 * CONSTRUCTOR: (Viele Defaults, kannst selber bestimmte dinge überschreiben)
 * @example constructor({memory, ...rest}: ChatBotProps = {}){
        this.memory = memory ?? new SmartCheckpointSaver(new MemorySaver())
        if ("chain" in rest){
            this.chain = rest.chain
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
 * @example oder eine chain wird anhand deines llms oder prompts erstellt mit memory:
 * @param props.memory
 * @param props.llm 
 * @param props.prompt 
 */
export class ChatBot {
    private memory: BaseCheckpointSaver
    private chain: Chain 

    constructor({memory, ...rest}: ChatBotProps = {}){
        this.memory = memory ?? new SmartCheckpointSaver(new MemorySaver())
        if ("chain" in rest){
            this.chain = rest.chain
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
        this.chain.addContext(data)
    }

    public async setContext(vectorStore: VectorStore){
        this.chain.setContext(vectorStore)
    }

    public async clearContext(){
        this.chain.clearContext()
    }

}

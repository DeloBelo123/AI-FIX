import { BaseCheckpointSaver, HumanMessage, AIMessage, LangGraphRunnableConfig, BaseMessage, CheckpointMetadata, VectorStore } from "../imports"
import { BaseChatModel } from "../imports"
import { SmartCheckpointSaver } from "../memory"
import { Chain } from "./chain"
import { MemorySaver } from "../imports"
import { getLLM } from "../helpers"
import { z } from "zod/v3"

type MemoryChainProps = { memory?: BaseCheckpointSaver } & ({
    chain: Chain
} | {
    llm?:BaseChatModel
    prompt?: Array<["system", string]>
    schema?: z.ZodObject<any, any>
})

/**
 * CONSTRUCTOR
 * @example  constructor({memory, ...rest}: MemoryChainProps = {}){
        this.memory = memory ?? new SmartCheckpointSaver(new MemorySaver())
        if ("chain" in rest){
            this.chain = rest.chain
        } else {
            this.chain = new Chain({
                llm: rest.llm ?? getLLM("groq"),
                prompt: rest.prompt ?? [["system", "Du bist ein hilfreicher Chatbot der mit dem User ein höffliches und hilfreiches Gespräch führt"]],
                schema: rest.schema
            })
        }
    }
 * 
 * @example entweder gibst du direkt eine chain + memory:
 * @param props.memory 
 * @param props.chain 
 * 
 * @example oder eine chain wird anhand deines llms oder prompts erstellt + memory:
 * @param props.memory
 * @param props.llm 
 * @param props.prompt 
 */
export class MemoryChain {
    private memory: BaseCheckpointSaver
    private chain: Chain<any>

    constructor({memory, ...rest}: MemoryChainProps = {}){
        this.memory = memory ?? new SmartCheckpointSaver(new MemorySaver())
        if ("chain" in rest){
            this.chain = rest.chain
        } else {
            this.chain = new Chain({
                llm: rest.llm ?? getLLM("groq"),
                prompt: rest.prompt ?? [["system", "Du bist ein hilfreicher Chatbot der mit dem User ein höffliches und hilfreiches Gespräch führt"]],
                schema: rest.schema
            })
        }
    }

    public async invoke(input: Record<string, any> & { thread_id: string, debug?: boolean }): Promise<any> {
        const config: LangGraphRunnableConfig = {
            configurable: { thread_id: input.thread_id }
        }
        
        const checkpoint = await this.memory.get(config)
    
        let historyMessages: BaseMessage[] = []
        if (checkpoint && checkpoint.channel_values && checkpoint.channel_values.messages) {
            historyMessages = checkpoint.channel_values.messages as BaseMessage[]
        }

        const historyText = historyMessages.map(msg => {
            if (msg instanceof HumanMessage) {
                return `User: ${typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)}`
            } else if (msg instanceof AIMessage) {
                return `Assistant: ${typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)}`
            } else {
                return `System: ${typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)}`
            }
        }).join('\n\n')

        const { thread_id, ...restInput } = input
        
        const invokeInput: Record<string, any> = {}
        for (const key in restInput) {
            if (key === "debug") continue
            invokeInput[key] = restInput[key]
        }
        
        if (historyText && invokeInput.message) {
            invokeInput.message = `${historyText}\n\nUser: ${invokeInput.message}`
        }

        try {
            const response = await this.chain.invoke(invokeInput)
            
            const responseText = typeof response === 'object' && 'output' in response 
                ? response.output 
                : typeof response === 'string' 
                ? response 
                : JSON.stringify(response)

            if (responseText && responseText.length > 0) {
                const userMessage = new HumanMessage(invokeInput.message || JSON.stringify(restInput))
                const aiMessage = new AIMessage(responseText)
                const newMessages = [...historyMessages, userMessage, aiMessage]
                
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

            return response
        } catch (error) {
            throw error
        }
    }

    public async *stream({input, thread_id}: {input: string, thread_id: string}): AsyncGenerator<string, string, unknown> {
        const config: LangGraphRunnableConfig = {
            configurable: { thread_id }
        }
        
        const checkpoint = await this.memory.get(config)
    
        let historyMessages: BaseMessage[] = []
        if (checkpoint && checkpoint.channel_values && checkpoint.channel_values.messages) {
            historyMessages = checkpoint.channel_values.messages as BaseMessage[]
        }

        const userMessage = new HumanMessage(input)
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
                message: `${historyText}\n\nUser: ${input}` 
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

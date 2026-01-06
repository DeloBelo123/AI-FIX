import { BaseCheckpointSaver, HumanMessage, AIMessage, LangGraphRunnableConfig, BaseMessage, CheckpointMetadata, VectorStore, DynamicStructuredTool } from "../imports"
import { BaseChatModel } from "../imports"
import { SmartCheckpointSaver } from "../memory"
import { MemorySaver } from "../imports"
import { logChunk } from "../helpers"
import { input } from "../cli"
import { MemoryChain } from "./memorychain"
import { Agent } from "./agent"

type ChatbotProps = {
    llm?: BaseChatModel
    prompt?: Array<["system", string]>
    tools?: DynamicStructuredTool[]
    memory?: BaseCheckpointSaver
}

export class Chatbot {
    private chain: MemoryChain | undefined
    private agent: Agent<any> | undefined

    constructor({llm,prompt,tools,memory}:ChatbotProps = {}){
        if(tools){
            this.agent = new Agent({
                memory: memory ?? new SmartCheckpointSaver(new MemorySaver()),
                tools : tools,
                prompt: prompt,
                llm: llm
            })
        } else {
            this.chain = new MemoryChain({
                memory: memory,
                prompt: prompt,
                llm: llm
            })
        }
    }

    public async *chat({input,thread_id}:{input:string, thread_id:string}): AsyncGenerator<string, string, unknown> {
        const streamable = this.chain ? this.chain : this.agent
        if (!streamable) throw new Error("kein streamable!")
        let fullResponse = ""
        for await (const chunk of streamable.stream({input,thread_id})){
            fullResponse += chunk
            yield chunk
        }
        return fullResponse
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
                    input: message, 
                    thread_id: id,
                })
                console.log("Assistant: ")
                for await (const chunk of response) {
                    logChunk(chunk)
                }
                console.log("") // Zeilenumbruch nach dem Streamen, damit Output nicht überschrieben wird
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
            await this.chain.addContext(data)
        } else if (this.agent) {
            await this.agent.addContext(data)
        } else {
            throw Error("weder agent noch chain, kein addContext möglich")
        }
    }

    public async setContext(vectorStore: VectorStore){
        if (this.chain){
            this.chain.setContext(vectorStore)
        } else if (this.agent) {
            this.agent.setContext(vectorStore)
        } else {
            throw Error("weder agent noch chain, kein setContext möglich")
        }
    }

    public async clearContext(){
        if (this.chain){
            this.chain.clearContext()
        } else if (this.agent) {
            this.agent.clearContext()
        } else {
            throw Error("weder agent noch chain, kein clearContext möglich")
        }
    }
}
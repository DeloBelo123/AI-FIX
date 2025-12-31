import { SupabaseTable } from "./supabase/server"
import { summarize, getLLM, Prettify } from "./helpers"
import {
    BaseCheckpointSaver,
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseChatModel,
    type Checkpoint,
    type CheckpointMetadata,
    type LangGraphRunnableConfig,
    type RunnableConfig,
    type CheckpointTuple,
    type CheckpointListOptions,
    type PendingWrite,
    type ChannelVersions
} from "./imports"

interface CheckpointRow {
    thread_id: string
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    created_at?: string
    updated_at?: string
}

export class SupabaseCheckpointSaver extends BaseCheckpointSaver {
    private table: SupabaseTable<CheckpointRow>
    
    constructor(supabaseTable: SupabaseTable<CheckpointRow>) {
        super()
        this.table = supabaseTable
    }
    
    async put(config: LangGraphRunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata): Promise<RunnableConfig> {
        const threadId = config.configurable?.thread_id || "default"
        const now = new Date().toISOString()
        
        // Prüfe ob Checkpoint bereits existiert um created_at zu behalten
        const existing = await this.table.select({
            columns: ["created_at"],
            where: [{ column: "thread_id", is: threadId }],
            first: true
        })
        
        // Behalte created_at wenn bereits vorhanden, sonst setze jetzt
        const createdAt = existing?.created_at || now
        
        await this.table.upsert({
            where: [{ column: "thread_id", is: threadId }],
            upsert: {
                checkpoint: checkpoint,
                metadata: metadata,
                created_at: createdAt,
                updated_at: now
            },
            onConflict: "thread_id"
        })
        
        return config
    }
    
    async get(config: LangGraphRunnableConfig): Promise<Checkpoint | undefined> {
        const threadId = config.configurable?.thread_id || "default"
        
        const data = await this.table.select({
            columns: ["checkpoint"],
            where: [{ column: "thread_id", is: threadId }],
            first: true
        })
        
        return data?.checkpoint
    }
    
    async *list(config: LangGraphRunnableConfig): AsyncGenerator<CheckpointTuple> {
        const threadId = config.configurable?.thread_id
        if (!threadId) return
        
        const data = await this.table.select({
            columns: ["checkpoint"],
            where: [{ column: "thread_id", is: threadId }]
        })
        
        for (const row of data || []) {
            yield [config, row.checkpoint] as unknown as CheckpointTuple
        }
    }
    
    async delete(config: LangGraphRunnableConfig): Promise<void> {
        const threadId = config.configurable?.thread_id || "default"
        
        await this.table.delete({
            where: [{ column: "thread_id", is: threadId }]
        })
    }
    
    async getTuple(config: LangGraphRunnableConfig): Promise<CheckpointTuple | undefined> {
        const checkpoint = await this.get(config)
        if (!checkpoint) return undefined
        return [config, checkpoint] as unknown as CheckpointTuple
    }
    
    async putWrites(config: RunnableConfig, writes: PendingWrite[], taskId: string): Promise<void> {
        for (const write of writes) {
            const checkpoint = (write as any).checkpoint as Checkpoint
            const metadata = (write as any).metadata as CheckpointMetadata
            await this.put(config as LangGraphRunnableConfig, checkpoint, metadata)
        }
    }
    
    async deleteThread(threadId: string): Promise<void> {
        await this.table.delete({
            where: [{ column: "thread_id", is: threadId }]
        })
    }
}

interface SmartCheckpointSaverOptions {
    messagesBeforeSummary?: number
    maxSummaries?: number
    llm?: BaseChatModel
    debug?: boolean
}
/**
 * @example CONSTRUCTOR:
 * constructor(
        checkpointSaver: BaseCheckpointSaver,
        {
            messagesBeforeSummary = 12,
            maxSummaries = 7,
            llm = getLLM("groq"),
            debug = false
        }: SmartCheckpointSaverOptions = {}
    ) {
        super()
        this.checkpointSaver = checkpointSaver
        this.messagesBeforeSummary = messagesBeforeSummary
        this.maxSummaries = maxSummaries
        this.llm = llm
        this.debug = debug
    }
 */
export class SmartCheckpointSaver extends BaseCheckpointSaver {
    private checkpointSaver: BaseCheckpointSaver
    private messagesBeforeSummary: number
    private maxSummaries: number
    private llm: BaseChatModel
    private debug: boolean

    constructor(
        checkpointSaver: BaseCheckpointSaver,{
            messagesBeforeSummary = 12,
            maxSummaries = 7,
            llm = getLLM("groq"),
            debug = false
        }: SmartCheckpointSaverOptions = {}
    ) {
        super()
        this.checkpointSaver = checkpointSaver
        this.messagesBeforeSummary = messagesBeforeSummary
        this.maxSummaries = maxSummaries
        this.llm = llm
        this.debug = debug
    }
    
    /**
     * Zählt User/AI Messages (ignoriert System-Messages)
     */
    private countChatMessages(messages: BaseMessage[]): number {
        return messages.filter(msg => 
            msg instanceof HumanMessage || msg instanceof AIMessage
        ).length
    }
    
    /**
     * Findet alle System-Messages die Zusammenfassungen sind
     */
    private findSummaryMessages(messages: BaseMessage[]): Array<{ index: number, message: SystemMessage }> {
        const summaries: Array<{ index: number, message: SystemMessage }> = []
        messages.forEach((msg, index) => {
            if (msg instanceof SystemMessage && 
                typeof msg.content === 'string' && 
                msg.content.includes('Zusammenfassung')) {
                summaries.push({ index, message: msg })
            }
        })
        return summaries
    }
    
    /**
     * Konvertiert Messages zu Text für Summarization
     */
    private messagesToText(messages: BaseMessage[]): string {
        return messages.map(msg => {
            const role = msg instanceof HumanMessage ? 'User' 
                : msg instanceof AIMessage ? 'Assistant' 
                : 'System'
            const content = typeof msg.content === 'string' 
                ? msg.content 
                : JSON.stringify(msg.content)
            return `${role}: ${content}`
        }).join('\n\n')
    }
    

    private async applySmartSummarization(checkpoint: Checkpoint): Promise<Checkpoint> {
        // Guard: Wenn checkpoint undefined ist, gib ihn zurück
        if (!checkpoint) {
            return checkpoint
        }
        
        // Messages sind in channel_values gespeichert
        const channelValues = checkpoint.channel_values || {}
        const messages = (channelValues.messages as BaseMessage[]) || []
        
        // Finde alle Zusammenfassungs-System-Messages
        const summaryMessages = this.findSummaryMessages(messages)
        
        // Finde den Index der letzten Zusammenfassung (falls vorhanden)
        const lastSummaryIndex = summaryMessages.length > 0 
            ? summaryMessages[summaryMessages.length - 1].index 
            : -1
        
        // Finde die Messages NACH der letzten Zusammenfassung
        const messagesAfterLastSummary = messages.slice(lastSummaryIndex + 1)
        
        // Zähle nur User/AI Messages NACH der letzten Zusammenfassung
        const chatMessageCount = this.countChatMessages(messagesAfterLastSummary)
        
        // Wenn noch nicht genug Messages nach der letzten Zusammenfassung, keine Summarization
        if (chatMessageCount < this.messagesBeforeSummary) {
            return checkpoint
        }
        
        // Finde die Indizes der letzten X User/AI Messages NACH der letzten Zusammenfassung die zusammengefasst werden sollen
        const indicesToSummarize: number[] = []
        const messagesToSummarize: BaseMessage[] = []
        let chatCount = 0
        
        // Gehe rückwärts durch Messages NACH der letzten Zusammenfassung
        for (let i = messagesAfterLastSummary.length - 1; i >= 0 && chatCount < this.messagesBeforeSummary; i--) {
            const msg = messagesAfterLastSummary[i]
            if (msg instanceof HumanMessage || msg instanceof AIMessage) {
                const originalIndex = lastSummaryIndex + 1 + i // Original-Index im messages Array
                indicesToSummarize.unshift(originalIndex) // Am Anfang einfügen für korrekte Reihenfolge
                messagesToSummarize.unshift(msg) // Am Anfang einfügen für korrekte Reihenfolge
                chatCount++
            }
        }
        
        if (messagesToSummarize.length === 0) {
            return checkpoint
        }
        
        // Erstelle Zusammenfassung
        const conversationText = this.messagesToText(messagesToSummarize)
        const summary = await summarize({
            conversation: conversationText,
            llm: this.llm,
            maxWords: 150
        })
        if (this.debug) {
            console.log(`Summary erstellt beim SmartCheckpointSaver: ${summary}`)
        }
        
        // Erstelle neue System-Message mit Zusammenfassung
        const summarySystemMessage = new SystemMessage(
            `Zusammenfassung der vorherigen Konversation:\n${summary}`
        )
        
        // Entferne die Messages die zusammengefasst wurden (verwende Indizes)
        const remainingMessages = messages.filter((_: BaseMessage, index: number) => 
            !indicesToSummarize.includes(index)
        )
        
        // Finde die Position der letzten Zusammenfassung im remainingMessages Array
        // (Die Indizes haben sich verschoben, aber die letzte Zusammenfassung sollte noch da sein)
        const remainingSummaryMessages = this.findSummaryMessages(remainingMessages)
        const lastSummaryIndexInRemaining = remainingSummaryMessages.length > 0 
            ? remainingSummaryMessages[remainingSummaryMessages.length - 1].index 
            : -1
        
        // Füge Zusammenfassung direkt nach der letzten Zusammenfassung ein
        // Wenn keine Zusammenfassung vorhanden, füge am Anfang ein (nach System-Messages)
        let newMessages: BaseMessage[]
        if (lastSummaryIndexInRemaining >= 0) {
            const beforeSummary = remainingMessages.slice(0, lastSummaryIndexInRemaining + 1)
            const afterSummary = remainingMessages.slice(lastSummaryIndexInRemaining + 1)
            newMessages = [...beforeSummary, summarySystemMessage, ...afterSummary]
        } else {
            // Keine Zusammenfassung vorhanden: Füge nach System-Messages ein
            const systemMessages = remainingMessages.filter((msg: BaseMessage) => msg instanceof SystemMessage)
            const nonSystemMessages = remainingMessages.filter((msg: BaseMessage) => !(msg instanceof SystemMessage))
            newMessages = [...systemMessages, summarySystemMessage, ...nonSystemMessages]
        }
        
        // Prüfe ob zu viele Zusammenfassungen vorhanden sind
        const allSummaries = this.findSummaryMessages(newMessages)
        if (allSummaries.length > this.maxSummaries) {
            // Entferne die älteste Zusammenfassung
            const oldestSummary = allSummaries[0]
            const finalMessages = newMessages.filter((_: BaseMessage, index: number) => index !== oldestSummary.index)
            
            return {
                ...checkpoint,
                channel_values: {
                    ...channelValues,
                    messages: finalMessages
                }
            }
        }
        
        return {
            ...checkpoint,
            channel_values: {
                ...channelValues,
                messages: newMessages
            }
        }
    }
    
    async put(config: LangGraphRunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, newVersions: ChannelVersions): Promise<RunnableConfig> {
        // Guard: Wenn checkpoint undefined ist, gib ihn direkt weiter
        if (!checkpoint) {
            return await this.checkpointSaver.put(config, checkpoint, metadata, newVersions)
        }
        
        // Wende Smart Summarization an
        const optimizedCheckpoint = await this.applySmartSummarization(checkpoint)
        
        // Speichere im unterliegenden CheckpointSaver
        return await this.checkpointSaver.put(config, optimizedCheckpoint, metadata, newVersions)
    }
    
    async get(config: LangGraphRunnableConfig): Promise<Checkpoint | undefined> {
        return await this.checkpointSaver.get(config)
    }
    
    async *list(config: LangGraphRunnableConfig, options?: CheckpointListOptions): AsyncGenerator<CheckpointTuple> {
        yield* this.checkpointSaver.list(config, options)
    }
    
    async delete(config: LangGraphRunnableConfig): Promise<void> {
        const threadId = config.configurable?.thread_id || "default"
        return await this.checkpointSaver.deleteThread(threadId)
    }
    
    async getTuple(config: LangGraphRunnableConfig): Promise<CheckpointTuple | undefined> {
        return await this.checkpointSaver.getTuple(config)
    }
    
    async putWrites(config: RunnableConfig, writes: PendingWrite[], taskId: string): Promise<void> {
        // putWrites wird für Channel-Updates verwendet, nicht für vollständige Checkpoints
        // Summarization wird nur in put() angewendet, wo vollständige Checkpoints gespeichert werden
        // Daher delegieren wir direkt an den unterliegenden Saver
        await this.checkpointSaver.putWrites(config, writes, taskId)
    }
    
    async deleteThread(threadId: string): Promise<void> {
        return await this.checkpointSaver.deleteThread(threadId)
    }
}
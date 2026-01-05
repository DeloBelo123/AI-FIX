import { ToolRegistry, type Tool } from "./BaseToolRegistry"
import { ZodiosToolRegistry } from "./ZodiosToolRegistry"
import { ZodiosEndpointWithAlias } from "./ZodiosToolRegistry"
import type { DynamicStructuredTool } from "@langchain/core/tools"
import { Zodios, type ApiOf } from "zodios"
import { z } from "zod/v3"

export type ExtractToolNames<T extends readonly {name:string}[]> = {
    [K in keyof T]: T[K] extends {name:string} ? T[K]['name'] : never
}[number]

type ExtractToolNamesFromZodiosApi<Api extends readonly any[]> = {
    [K in keyof Api]: Api[K] extends {name: infer N}
        ? N extends string
            ? N
            : Api[K] extends {method: infer M, path: infer P}
            ? `call api ${M & string} ${P & string}`
            : never
        : Api[K] extends {method: infer M, path: infer P}
        ? `call api ${M & string} ${P & string}`
        : never
}[number]

type ExtractToolNamesFromInput<T extends readonly (Tool | Zodios<any>)[]> = {
    [K in keyof T]: T[K] extends Tool
        ? T[K]['name']
        : T[K] extends Zodios<infer Api>
        ? ExtractToolNamesFromZodiosApi<ApiOf<T[K]>>
        : never
}[number]

/**
 * vergiss nicht 'as const' am ende des input-arrays zu verwenden für perekten autocomplete!!!
 */
export class CombinedToolRegistry<T extends readonly (Tool | Zodios<any>)[]> {
    private BaseToolRegistry:ToolRegistry<Tool[]> | undefined
    private ZodiosToolRegistry:ZodiosToolRegistry<Zodios<any>> | undefined
    private allTools:DynamicStructuredTool[]

    constructor(input: T){
        //getting and checking the tools
        const baseTools = input.filter((tool)=>{
            return typeof tool === "object" && tool !== null && "name" in tool && "description" in tool && "schema" in tool && "func" in tool
        })
        if(baseTools.length > 0){
            this.BaseToolRegistry = new ToolRegistry(baseTools)
        }
        
        //getting and checking the zodios clients
        const zodiosClients = input.filter((item) => {
            return item instanceof Zodios
        })
        if(zodiosClients.length > 1){
            throw new Error("Only one Zodios client can be registered")
        }
        const zodiosClient = zodiosClients[0]
        if (zodiosClient) {
            this.ZodiosToolRegistry = new ZodiosToolRegistry(zodiosClient)
        }

        //combining the tools
        this.allTools = [...(this.BaseToolRegistry?.Tools || []), ...(this.ZodiosToolRegistry?.Tools || [])]
    }

    getTool(name: ExtractToolNamesFromInput<T>): DynamicStructuredTool | undefined {
        return this.allTools.find((tool) => tool.name === name)
    }

    getTools(...names: ExtractToolNamesFromInput<T>[]): DynamicStructuredTool[] {
        return names.map((name) => this.getTool(name)).filter((tool): tool is DynamicStructuredTool => tool !== undefined)
    }

    get Tools(): DynamicStructuredTool[] {
        return this.allTools
    }
}


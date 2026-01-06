import type { Zodios, ZodiosEndpointDescription, ApiOf } from "zodios"
import { DynamicStructuredTool } from "../../imports"
import { z, type ZodTypeAny } from 'zod/v3'
import { Prettify } from "../../helpers"
import type { ExtractToolNames } from "./CombinedRegistry"

export type ZodiosEndpointWithAlias<R> = Prettify<ZodiosEndpointDescription<R> & {
    name?: string
}>

export class ZodiosToolRegistry<Z extends Zodios<any>> {
    private apiSchema: ApiOf<Z>
    private tools: DynamicStructuredTool[] 
    private zodiosClient: Z
    constructor(zodiosClient: Z){
        this.zodiosClient = zodiosClient
        this.apiSchema = (zodiosClient as any).api as ApiOf<Z>
        this.tools = this.turnApiIntoTools()
    }
    
    public getTool(name: ExtractToolNames<ApiOf<Z>>): DynamicStructuredTool | undefined {
        const tools = this.tools.filter((tool) => tool.name?.toLowerCase() === name.toLowerCase())
        if (tools.length > 1){
            throw new Error(`Error! es wurden unter dem gleichen namen ${name} mehrere tools registriert!`)
        }
        const tool = tools[0]
        if(!tool){
            console.error(`Tool ${name} not found`)
            return undefined
        }
        return tool
    }

    public getTools(...names: ExtractToolNames<ApiOf<Z>>[]): (DynamicStructuredTool | undefined)[] {
        return names.map(name => this.getTool(name))
    }

    get allTools(): DynamicStructuredTool[] {
        return this.tools
    }

    private turnApiIntoTools():DynamicStructuredTool[]{
        return this.apiSchema.map((endpoint: ZodiosEndpointDescription<any> & { name?: string })=>{
            return new DynamicStructuredTool({
                name:endpoint.name || `call api ${endpoint.method} ${endpoint.path}`,
                description: endpoint.description|| `calls the api ${endpoint.method} ${endpoint.path}`,
                schema:this.buildToolSchema(endpoint),
                func: async (input) => {
                    return this.zodiosClient.request({
                      method: endpoint.method,
                      url: endpoint.path,
                      params: input?.params,
                      queries: input?.queries,
                      headers: input?.headers,
                      data: input?.body,
                    } as any)
                },
            })
        })
    }

    private buildToolSchema(endpoint: ZodiosEndpointDescription<any>) {
        const queries: Record<string, ZodTypeAny> = {}
        const headers: Record<string, ZodTypeAny> = {}
        let body: ZodTypeAny | undefined

        for (const param of endpoint.parameters ?? []) {
            if (param.type === "Query") {
            queries[param.name] = param.schema as unknown as ZodTypeAny
            }
            if (param.type === "Header") {
            headers[param.name] = param.schema as unknown as ZodTypeAny
            }
            if (param.type === "Body") {
            body = param.schema as unknown as ZodTypeAny
            }
        }

        const schemaShape: Record<string, ZodTypeAny> = {}
        
        if (Object.keys(queries).length > 0) {
            schemaShape.queries = z.object(queries).optional()
        }
        
        if (Object.keys(headers).length > 0) {
            schemaShape.headers = z.object(headers).optional()
        }
        
        if (body) {
            schemaShape.body = body.optional()
        }

        return z.object(schemaShape)
    }

}
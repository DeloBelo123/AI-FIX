import { DynamicStructuredTool } from "@langchain/core/tools"
import { z } from "zod/v3"

interface Tool {
    name:string
    description:string
    schema:z.AnyZodObject
    func:(...args:any[]) => any
}

export class ToolRegistry {
    private tools:DynamicStructuredTool[]
    constructor(tools:Array<Tool | DynamicStructuredTool>){
        this.tools = tools.map(tool => tool instanceof DynamicStructuredTool ? tool : this.turnToTool(tool))
        if (this.checkDuplicatedTools()){
            throw new Error(`Error! mehrere tools wurden unter den gleichen Namen registriert!`)
        }
    }

    public getTool(name:string): DynamicStructuredTool | undefined{
        const tools = this.tools.filter(tool => tool.name.toLowerCase() === name.toLowerCase())
        if (tools.length > 1) {
            throw new Error(`Error! mehrere tools wurden unter den gleichen Namen ${name} registriert!`)
        }
        if (tools.length !== 1){
            console.error(`unter ${name} wurde kein Tool gefunden`)
            return
        }
        return tools[0]
    }

    public getTools(names:string[]){
        return names.map(name => this.getTool(name))
    }

    public addTool(tool:DynamicStructuredTool | Tool){
        for(let i = 0; i < this.tools.length; i++){
            if(this.tools[i].name === tool.name){
                throw new Error(`Error! tool ${tool.name} wurde bereits registriert!`)
            }
        }
        if (tool instanceof DynamicStructuredTool){
            this.tools.push(tool)
        } else {
            this.tools.push(new DynamicStructuredTool({
                name:tool.name,
                description:tool.description,
                schema:tool.schema,
                func:tool.func
            }))
        }
    }

    public deleteTool(name:string){
        for(let i = 0; i < this.tools.length; i++){
            if(this.tools[i].name === name){
                const name_to_make_sure = this.tools.splice(i,1)
                console.log(`actualy deleted ${name_to_make_sure}. for reference, this tool was given by the param:${name}`)
            }
        }
    }

    public updateTool(name:string,tool:DynamicStructuredTool){
        if(name !== tool.name){
            throw new Error(`man, was labberst du da? wie willst du ${name} updaten wenn du ${tool.name} als tool eingibst? die haben nicht den gleichen Namen!!!`)
        }
        this.deleteTool(name)
        this.addTool(tool)
    }

    private turnToTool(tool:Tool):DynamicStructuredTool{
        return new DynamicStructuredTool({
            name:tool.name,
            description:tool.description,
            schema:tool.schema,
            func:tool.func
        })

    }

    private checkDuplicatedTools():boolean{
        const dublikaes = this.tools.filter((tool,index)=>{
            return this.tools.indexOf(tool) !== index
        })
        return dublikaes.length > 0 ? true : false
    }

    public get Tools():DynamicStructuredTool[]{
        return this.tools
    }
}

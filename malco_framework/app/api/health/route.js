import {config} from "../../../lib/runtime.js";
export const dynamic="force-dynamic";
export function GET(){const c=config();return Response.json({status:"ok",mode:c.mode,tasks:c.tasks});}

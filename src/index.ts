// 全局前置配置：禁用 multipart 上传 + 兜底异常捕获
process.env.LANGSMITH_DISABLE_MULTIPART = "true";
process.env.LANGSMITH_TRACING = "true";

// 优先加载环境变量
import "dotenv/config";

// 劫持 LangSmith 上报方法，静默忽略 403 错误
import { Client } from "langsmith";
const originalIngest = Client.prototype._ingestRuns;
Client.prototype._ingestRuns = async function (...args: unknown[]) {
  try {
    return await originalIngest.apply(this, args);
  } catch (err) {
    // 仅忽略 403 上报错误，其它异常正常抛出
    const msg = (err as Error)?.message ?? "";
    if (msg.includes("403") || msg.includes("multipart request")) {
      return;
    }
    throw err;
  }
};

// 业务依赖导入
import { createAgent, tool } from "langchain";
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import "langsmith";

// 定义天气查询工具
const getWeather = tool(
  ({ city }) => `${city} 今日天气：晴，气温22°C，湿度60%`,
  {
    name: "get_weather",
    description: "查询指定城市的当前天气",
    schema: z.object({
      city: z.string().describe("要查询天气的城市名称"),
    }),
  }
);

// 校验千问环境变量
const apiKey = process.env.QIANWEN_API_KEY;
const baseUrl = process.env.QWEN_API_URL;

if (!apiKey || !baseUrl) {
  throw new Error("环境变量缺失，请检查 .env 文件");
}

// 初始化千问模型
const model = new ChatOpenAI({
  model: "qwen-turbo",
  apiKey,
  streaming: false,
  configuration: {
    baseURL: baseUrl,
  },
});

// 创建 Agent
const agent = createAgent({
  model,
  tools: [getWeather],
});

// 执行调用并输出结果
const result = await agent.invoke({
  messages: [{ role: "user", content: "北京和上海今天天气怎么样？" }],
});

console.log(result.messages.at(-1)?.content);
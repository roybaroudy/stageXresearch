import { useState, useRef, useEffect } from "react";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { ScrollArea } from "@/components/ui/scroll-area";

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

interface ChatInterfaceProps {
  initialMessage?: string;
}

export const ChatInterface = ({ initialMessage }: ChatInterfaceProps) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // Simulate API response
  const generateResponse = async (userMessage: string): Promise<string> => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
    
    // Mock responses about ISO/IEC 42001
    const responses = [
      "ISO/IEC 42001 is an international standard that provides a framework for establishing, implementing, maintaining, and continually improving an AI management system. It helps organizations manage AI risks and opportunities systematically.",
      "The standard covers key areas including AI governance, risk management, data quality, transparency, and accountability. It's designed to be applicable to any organization using or developing AI systems.",
      "One of the key benefits of ISO/IEC 42001 is that it provides a structured approach to AI governance, helping organizations build trust with stakeholders and demonstrate responsible AI practices.",
      "The standard emphasizes the importance of continuous monitoring and improvement of AI systems, ensuring they remain aligned with organizational objectives and ethical principles throughout their lifecycle.",
      "ISO/IEC 42001 also addresses the need for proper documentation, training, and competence management in AI-related activities, helping organizations build internal capabilities for responsible AI management."
    ];
    
    return responses[Math.floor(Math.random() * responses.length)];
  };

  const handleSendMessage = async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await generateResponse(content);
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "I apologize, but I'm unable to provide a response at the moment. Please try again."
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle initial message
  useEffect(() => {
    if (initialMessage) {
      handleSendMessage(initialMessage);
    }
  }, [initialMessage]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollElement = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollElement) {
        scrollElement.scrollTop = scrollElement.scrollHeight;
      }
    }
  }, [messages, isLoading]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex-shrink-0 p-6 border-b border-border bg-card/50">
        <h1 className="text-2xl font-semibold text-foreground">ISO/IEC 42001 Assistant</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Ask me anything about the AI Management System standard
        </p>
      </div>

      {/* Messages */}
      <div className="flex-1 relative">
        <ScrollArea className="h-full" ref={scrollAreaRef}>
          <div className="p-6 space-y-4">
            {messages.length === 0 && !isLoading && (
              <div className="text-center py-12">
                <p className="text-muted-foreground">
                  Start by asking a question about ISO/IEC 42001
                </p>
              </div>
            )}
            
            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                role={message.role}
                content={message.content}
              />
            ))}
            
            {isLoading && (
              <ChatMessage
                role="assistant"
                content=""
                isTyping={true}
              />
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Input */}
      <div className="flex-shrink-0 p-6 border-t border-border bg-card/50">
        <ChatInput
          onSendMessage={handleSendMessage}
          disabled={isLoading}
          placeholder="Ask about ISO/IEC 42001 requirements, implementation, benefits..."
        />
      </div>
    </div>
  );
};
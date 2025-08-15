import { useState } from "react";
import { ChatInterface } from "@/components/ChatInterface";
import { ChatInput } from "@/components/ChatInput";
import { Navigation } from "@/components/Navigation";
import { Card } from "@/components/ui/card";

const Index = () => {
  const [hasStarted, setHasStarted] = useState(false);
  const [initialMessage, setInitialMessage] = useState<string>();

  const handleStartChat = (message: string) => {
    setInitialMessage(message);
    setHasStarted(true);
  };

  if (hasStarted) {
    return <ChatInterface initialMessage={initialMessage} />;
  }

  return (
    <div className="min-h-screen bg-gradient-chat">
      <Navigation />
      <div className="flex items-center justify-center p-4" style={{ minHeight: 'calc(100vh - 4rem)' }}>
        <div className="w-full max-w-2xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-4">
            ISO/IEC 42001
          </h1>
          <h2 className="text-xl md:text-2xl text-primary font-semibold mb-4">
            AI Management System Standard
          </h2>
          <p className="text-muted-foreground text-lg max-w-lg mx-auto">
            Get instant answers about ISO/IEC 42001 requirements, implementation guidance, and best practices for AI governance.
          </p>
        </div>

        <Card className="p-8 shadow-medium border-border/50 bg-card/80 backdrop-blur-sm">
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-foreground mb-2">
              Ask your first question
            </h3>
            <p className="text-sm text-muted-foreground">
              Example: "What are the key requirements of ISO/IEC 42001?" or "How do I implement an AI management system?"
            </p>
          </div>
          
          <ChatInput
            onSendMessage={handleStartChat}
            placeholder="What would you like to know about ISO/IEC 42001?"
            className="w-full"
          />
        </Card>

        <div className="mt-8 text-center">
          <p className="text-sm text-muted-foreground">
            Powered by advanced RAG technology for accurate, up-to-date information
          </p>
        </div>
        </div>
      </div>
    </div>
  );
};

export default Index;

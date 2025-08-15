import { Navigation } from "@/components/Navigation";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Shield, Users, FileText, Target, Zap, CheckCircle } from "lucide-react";

const About = () => {
  return (
    <div className="min-h-screen bg-gradient-chat">
      <Navigation />
      
      <div className="container mx-auto max-w-4xl px-4 py-12">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-foreground mb-4">
            About ISO/IEC 42001 Assistant
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Your intelligent companion for understanding and implementing the AI Management System standard
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-2 mb-12">
          <Card className="p-6 shadow-medium">
            <div className="flex items-center space-x-3 mb-4">
              <Shield className="h-8 w-8 text-primary" />
              <h2 className="text-2xl font-semibold">What is ISO/IEC 42001?</h2>
            </div>
            <p className="text-muted-foreground leading-relaxed">
              ISO/IEC 42001 is the world's first international standard for AI management systems. 
              It provides a framework for organizations to develop, deploy, and use AI responsibly 
              and ethically while managing associated risks and opportunities.
            </p>
          </Card>

          <Card className="p-6 shadow-medium">
            <div className="flex items-center space-x-3 mb-4">
              <Zap className="h-8 w-8 text-primary" />
              <h2 className="text-2xl font-semibold">Why This Assistant?</h2>
            </div>
            <p className="text-muted-foreground leading-relaxed">
              This RAG-powered assistant provides instant, accurate answers about ISO/IEC 42001 
              requirements, implementation strategies, and best practices, making the complex 
              standard accessible and actionable for your organization.
            </p>
          </Card>
        </div>

        <Card className="p-8 shadow-medium mb-12">
          <h2 className="text-2xl font-semibold mb-6 text-center">Key Areas Covered</h2>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="flex items-center space-x-3">
              <CheckCircle className="h-5 w-5 text-primary flex-shrink-0" />
              <span className="text-sm">AI Governance Framework</span>
            </div>
            <div className="flex items-center space-x-3">
              <CheckCircle className="h-5 w-5 text-primary flex-shrink-0" />
              <span className="text-sm">Risk Management</span>
            </div>
            <div className="flex items-center space-x-3">
              <CheckCircle className="h-5 w-5 text-primary flex-shrink-0" />
              <span className="text-sm">Data Quality Assurance</span>
            </div>
            <div className="flex items-center space-x-3">
              <CheckCircle className="h-5 w-5 text-primary flex-shrink-0" />
              <span className="text-sm">Transparency & Accountability</span>
            </div>
            <div className="flex items-center space-x-3">
              <CheckCircle className="h-5 w-5 text-primary flex-shrink-0" />
              <span className="text-sm">Continuous Monitoring</span>
            </div>
            <div className="flex items-center space-x-3">
              <CheckCircle className="h-5 w-5 text-primary flex-shrink-0" />
              <span className="text-sm">Stakeholder Engagement</span>
            </div>
          </div>
        </Card>

        <div className="grid gap-6 md:grid-cols-3">
          <Card className="p-6 text-center shadow-medium">
            <Users className="h-12 w-12 text-primary mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">For Organizations</h3>
            <p className="text-sm text-muted-foreground">
              Implement AI governance frameworks that build stakeholder trust and ensure responsible AI deployment.
            </p>
          </Card>

          <Card className="p-6 text-center shadow-medium">
            <FileText className="h-12 w-12 text-primary mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">For Consultants</h3>
            <p className="text-sm text-muted-foreground">
              Access comprehensive guidance to help clients navigate ISO/IEC 42001 requirements and implementation.
            </p>
          </Card>

          <Card className="p-6 text-center shadow-medium">
            <Target className="h-12 w-12 text-primary mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">For Auditors</h3>
            <p className="text-sm text-muted-foreground">
              Understand audit criteria and assessment methodologies for AI management system compliance.
            </p>
          </Card>
        </div>

        <div className="text-center mt-12">
          <div className="flex justify-center space-x-2 mb-4">
            <Badge variant="secondary">RAG Technology</Badge>
            <Badge variant="secondary">Real-time Responses</Badge>
            <Badge variant="secondary">Expert Knowledge</Badge>
          </div>
          <p className="text-sm text-muted-foreground">
            Powered by advanced Retrieval-Augmented Generation for accurate, contextual answers
          </p>
        </div>
      </div>
    </div>
  );
};

export default About;
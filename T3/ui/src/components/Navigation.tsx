import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Info } from "lucide-react";

export const Navigation = () => {
  const location = useLocation();
  
  return (
    <nav className="bg-card/90 backdrop-blur-sm border-b border-border sticky top-0 z-50">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo/Home Button */}
          <Link to="/" className="hover:opacity-80 transition-opacity">
            <div>
              <h1 className="text-xl font-bold text-foreground">ISO/IEC 42001</h1>
              <p className="text-xs text-muted-foreground -mt-1">AI Management Assistant</p>
            </div>
          </Link>

          {/* Navigation Links */}
          <div className="flex items-center space-x-4">
            <Button
              variant={location.pathname === '/about' ? 'default' : 'ghost'}
              size="sm"
              asChild
              className="flex items-center space-x-2"
            >
              <Link to="/about">
                <Info className="h-4 w-4" />
                <span>About</span>
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </nav>
  );
};
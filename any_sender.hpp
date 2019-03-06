#pragma once

#include <utility>
#include <memory>

// a type eraser for senders
class any_sender
{
  private:
    struct sender
    {
      virtual ~sender() = default;
      virtual cudaGraphNode_t insert(cudaGraph_t g) const = 0;
      virtual void submit() = 0;
      virtual void sync_wait() const = 0;
    };

    template<class Sender>
    struct erased_sender : sender
    {
      erased_sender(Sender&& sender)
        : sender_(std::move(sender))
      {}

      virtual cudaGraphNode_t insert(cudaGraph_t g) const
      {
        return sender_.insert(g);
      }

      virtual void submit()
      {
        sender_.submit();
      }

      virtual void sync_wait() const
      {
        sender_.sync_wait();
      }

      Sender sender_;
    };

    std::unique_ptr<sender> sender_;

  public:
    any_sender() = default;
    any_sender(any_sender&&) = default;
    any_sender(const any_sender&) = delete;

    template<class Sender>
    any_sender(Sender&& sender)
      : sender_(new erased_sender<typename std::decay<Sender>::type>(std::forward<Sender>(sender)))
    {}

    cudaGraphNode_t insert(cudaGraph_t g) const
    {
      if(!sender_)
      {
        throw std::runtime_error("any_sender::insert: invalid state.");
      }

      return sender_->insert(g);
    }

    void submit()
    {
      if(!sender_)
      {
        throw std::runtime_error("any_sender::submit: invalid state.");
      }

      sender_->submit();
    }

    void sync_wait() const
    {
      if(!sender_)
      {
        throw std::runtime_error("any_sender::sync_wait: invalid state.");
      }

      sender_->sync_wait();
    }
};


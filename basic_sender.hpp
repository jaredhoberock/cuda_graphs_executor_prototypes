#pragma once

template<class Executor, class Function, class Sender>
class basic_sender
{
  public:
    const Executor& executor() const
    {
      return executor_;
    }

    // XXX consider adding this method if we can do so generically
    //void submit();

    // XXX consider adding this method if we can do so generically
    //void sync_wait() const;

  protected:
    basic_sender(const Executor& executor, Function function, Sender&& predecessor)
      : executor_(executor),
        function_(function),
        predecessor_(std::move(predecessor))
    {}

    const Function& function() const
    {
      return function_;
    }

    const Sender& predecessor() const
    {
      return predecessor_;
    }

  private:
    Executor executor_;
    Function function_;
    Sender predecessor_;
};

